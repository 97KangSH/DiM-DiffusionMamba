# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from collections import deque

from timm.models.layers import DropPath

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, mamba_inner_ref, mamba_inner_fn_no_out_proj

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from copy import deepcopy


class HTM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        num_module=1, # value k
        aggregate='concat_linear', # 'sum', 'concat_linear'
        log=False,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.num_module = num_module
        self.aggregate = aggregate
        self.log = log
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # HTM block convolution initialization - by KANG
        conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        # HTM block deep copy - by KANG
        if log:
            print(num_module)
        self.conv1ds = nn.ModuleList([deepcopy(conv1d) for k in range(num_module)])

        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
        self.dt_projs = nn.ModuleList()
        for k in range(num_module):
            self.dt_projs.append(nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs))

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        for k in range(num_module):
            if dt_init == "constant":
                nn.init.constant_(self.dt_projs[k].weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_projs[k].weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        for k in range(num_module):
            with torch.no_grad():
                self.dt_projs[k].bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            self.dt_projs[k].bias._no_reinit = True

        # S4D real initialization
        # k number S4D initialization - by KANG
        self.A_logs = []
        self.Ds = []
        
        for k in range(num_module):
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_log = torch.log(A)  # Keep A_log in fp32
            self.A_logs.append(nn.Parameter(A_log))
            self.A_logs[k]._no_weight_decay = True

            # D "skip" parameter
            self.Ds.append(nn.Parameter(torch.ones(self.d_inner, device=device)))  # Keep in fp32
            self.Ds[k]._no_weight_decay = True
        
        # aggregate
        if self.aggregate == 'concat_linear':
            self.aggregate_linear = nn.Linear(self.d_inner * self.num_module, self.d_inner, bias = False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
    def forward(self, hidden_states, inference_params=None):
        """

        hidden_states: (B, T, D)
        Returns: same shape as hidden_states
        """
        
        batch, seqlen, dim = hidden_states.shape
        if self.log:
            print('##### HTM_Input.shape #####')
            print(hidden_states.shape)
            print(f"torch.Size([B,  T,  D])")
        # print(f'B: {batch}, T: {seqlen} D: {dim}')
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.log:
            print('##### xz.shape #####')
            print(xz.shape)
            print(f"torch.Size([B,E*expand*2,  T])")
            print(f"E_expand: {xz.shape[1]//2}")
        # print(f'B: {xz.shape[0]}, E*2: {xz.shape[1]}, L/T: {xz.shape[2]} ')
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        # print('xz.shape', xz.shape)

        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            outputs = []
            for dt_proj, conv1d, A_log, D in zip(self.dt_projs, self.conv1ds, self.A_logs, self.Ds):
                A = -torch.exp(A_log.float()) 
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    conv1d.weight,
                    conv1d.bias,
                    self.x_proj.weight,
                    dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    D.float(),
                    delta_bias=dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out = rearrange(out, "b d l -> b l d")
                outputs.append(out)
            if self.log:
                print('##### y.shape #####')
                print(outputs[0].shape)
                print(f"torch.Size([B,  T,E*expand*2])")

            # print(f'B: {outputs[0].shape[0]}, L/T: {outputs[0].shape[1]}, E: {outputs[0].shape[2]} ')
            if self.aggregate == 'sum':
                if self.log:
                    print('sum')
                aggregated = torch.stack(outputs, dim=0).sum(dim=0)
            elif self.aggregate == 'concat_linear':
                aggregated = torch.cat(outputs, dim=-1)
                if self.log:
                    print('##### concat_linear #####')
                    print(aggregated.shape)
                    print(f"torch.Size([B,  T,E*{self.num_module}])")
                aggregated = self.aggregate_linear(aggregated)
            
            out = self.out_proj(aggregated)

            if self.log:
                print('##### out.shape #####')
                print(out.shape)
                print(f"torch.Size([B,  T,  D])")
            return out
        else:
            x, z = xz.chunk(2, dim=1)
            if self.log:
                print('x.shape',x.shape)
                print('z.shape',z.shape)
                print(f'B: {x.shape[0]}, E: {x.shape[1]}, L/T: {x.shape[2]} ')
            outputs = []
            for dt_proj, conv1d, A_log, D in zip(self.dt_projs, self.conv1ds, self.A_logs, self.Ds):
                A = -torch.exp(A_log.float())  # (d_inner, d_state)
        
                # Compute short convolution
                if conv_state is not None:
                    # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
                if causal_conv1d_fn is None:
                    x = self.act(conv1d(x)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x=x,
                        weight=rearrange(conv1d.weight, "d 1 w -> d w"),
                        bias=conv1d.bias,
                        activation=self.activation,
                    )
                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
                x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = dt_proj.weight @ dt.t()
                # Paper: N, code: d_state
                # 캬 l=seqlen 이걸로 seq 길이 정상화하는구나~!!
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D.float(),
                    z=z,
                    delta_bias=dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b d l -> b l d")
                outputs.append(y)

            if self.log:
                print('y.shape: ', outputs[0].shape)
                print(f'B: {outputs[0].shape[0]}, L/T: {outputs[0].shape[1]}, E: {outputs[0].shape[2]} ')
            if self.aggregate == 'sum':
                if self.log:
                    print('sum')
                aggregated = torch.stack(outputs, dim=0).sum(dim=0)
            elif self.aggregate == 'concat_linear':
                
                aggregated = torch.cat(outputs, dim=-1)
                if self.log:
                    print('concat_linear')
                    print(aggregated.shape)
                
                aggregated = self.aggregate_linear(aggregated)
            
            out = self.out_proj(aggregated)

            if self.log:
                print('out.shape: ', out.shape)
                print(f'B: {out.shape[0]}, L/T: {out.shape[1]}, E: {out.shape[2]} ')
            return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
    
    # @override
    def to(self, *args, **kwargs):
        # 기존 `to` 메서드 호출
        super_result = super().to(*args, **kwargs)
        
        # 추가 기능
        if "cuda" in str(args[0]):  # GPU로 이동하는 경우 추가 작업 수행
            for k in range(self.num_module):
                self.A_logs[k] = self.A_logs[k].to("cuda")
                self.Ds[k] = self.Ds[k].to("cuda")

        return super_result


class BSM(nn.Module):
    def __init__(
        self,
        d_temporal=2,     # latent [2, 256] 에서 2를 맡고 있는 녀석이다.
        d_model=256,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5, 
        rms_norm: bool = True, 
        residual_in_fp32: bool = True,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        if_divide_out=False,
        init_layer_scale=None,
        log=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_temporal = d_temporal
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_temporal)
        self.dt_rank = math.ceil(self.d_temporal / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.if_divide_out = if_divide_out
        self.residual_in_fp32 = residual_in_fp32
        self.init_layer_scale = init_layer_scale
        self.log = log

        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((self.d_temporal)), requires_grad=True)

        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(
            self.d_temporal, eps=norm_epsilon, **factory_kwargs
        )
        self.in_proj = nn.Linear(self.d_temporal, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True 

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_temporal, bias=bias, **factory_kwargs)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            self.d_temporal, eps=norm_epsilon, **factory_kwargs
        )

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, T, D)
        Returns: same shape as hidden_states
        """

        
        if self.log:
            print("##### BSM_Input.shae#####")
            print(hidden_states.shape)
            print(f"torch.Size([B,  T,  D])")

        hidden_states = rearrange(hidden_states, "b l d -> b d l")
        batch, seqlen, dim = hidden_states.shape
        if self.log:
            print("##### Rearrange.shape#####")
            print(hidden_states.shape)
            print(f"torch.Size([B,  D,  T])")

        init_fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
        # residual: hidden_state
        hidden_states, residual = init_fused_add_norm_fn(
            hidden_states,
            self.norm.weight,
            self.norm.bias,
            residual=None,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
            eps=self.norm.eps,
        )
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        # xz = rearrange(
        #     self.in_proj.weight @ rearrange(hidden_states, "b d l -> l (b d)"),
        #     "l (b d) -> b l d",
        #     l=seqlen,
        # )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            # xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "l -> l 1")
        
        if self.log:
            print('##### xz.shape #####')
            print(xz.shape)
            print(f"torch.Size([B,T*expand*2,  E])")
            print(f"T_expand: {xz.shape[1]//2}")
        # print("[B, T*2, E]가 되어야 함 (아직)")
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            A_b = -torch.exp(self.A_b_log.float())
            out = mamba_inner_fn_no_out_proj(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            if self.log:
                print('##### forward.shape #####')
                print(out.shape)
                print(f"torch.Size([B,  T,  E])")
            out_b = mamba_inner_fn_no_out_proj(
                xz.flip([-1]),
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj_b.weight,
                self.dt_proj_b.weight,
                A_b,
                None,
                None,
                self.D_b.float(),
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            )
            if self.log:
                print('##### backward.shape #####')
                print(out.shape)
                print(f"torch.Size([B,  T,  E ])")
            # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
            if not self.if_divide_out:
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight, self.out_proj.bias)
        else:

            ## TODO: self.use_fast_path가 아닐 때 수행하는 경로 작성하기.
            
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        
        if self.log:
            print("##### Output shape #####")
            print(out.shape)
            print(f"torch.Size([B,  D,  T])")

        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        out = fused_add_norm_fn(
            self.drop_path(out),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
        )
        if self.log:
            print("##### After Add residual #####")
            print(out.shape)
            print(f"torch.Size([B,  D,  T])")

        out = rearrange(out, "b d l -> b l d")
        if self.log:
            print("##### Final_Result #####")
            print(out.shape)
            print(f"torch.Size([B,  T,  D])")

        if self.init_layer_scale is not None:
                out = out * self.gamma    
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_temporal * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_temporal * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_temporal * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_temporal * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class MotionMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_temporal,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5, 
        rms_norm: bool = True, 
        residual_in_fp32: bool = True,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        num_module=1, # value k
        if_divide_out=False,
        init_layer_scale=None,
        aggregate='concat_linear', # 'sum', 'concat_linear'
        log = False,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.num_module = num_module
        self.aggregate = aggregate
        self.log = log

        self.gate = nn.Linear(
                self.d_model, self.d_model, bias=True, **factory_kwargs
            )
        
        self.htm = HTM(d_model,
                       d_state,
                       d_conv,
                       expand,
                       dt_rank,
                       dt_min,
                       dt_max,
                       dt_init,
                       dt_scale,
                       dt_init_floor,
                       conv_bias,
                       bias,
                       use_fast_path,
                       layer_idx,
                       device,
                       dtype,
                       num_module,
                       aggregate,
                       log,
                       **kwargs)
        
        self.bsm = BSM(d_temporal,
                       d_model,
                       d_state,
                       d_conv,
                       expand,
                       dt_rank,
                       dt_min,
                       dt_max,
                       dt_init,
                       dt_scale,
                       dt_init_floor,
                       drop_rate,
                       drop_path_rate,
                       norm_epsilon,
                       rms_norm,
                       residual_in_fp32,
                       conv_bias,
                       bias,
                       use_fast_path,
                       layer_idx,
                       device,
                       dtype,
                       if_divide_out,
                       init_layer_scale,
                       log)
        
    def forward(self, hidden_states, inference_params=None):
        """

        hidden_states: (B, T, D)
        Returns: same shape as hidden_states
        """
        
        batch, seqlen, dim = hidden_states.shape
        if self.log:
            print('##### Mamba Block Input.shape #####')
            print(hidden_states.shape)
            print(f"torch.Size([B,  T,  D])")
        # print(f'B: {batch}, T: {seqlen} D: {dim}')
        gate = self.gate(hidden_states)
        if self.log:
            print("##### Gate.shape #####")
            print(gate.shape)
            print(f"torch.Size([B,  T,  D])")
        
        hidden_states = self.htm(hidden_states)
        hidden_states = self.bsm(hidden_states)
        
        out = hidden_states * gate
        if self.log:
            print("##### Final Output.shape #####")
            print(out.shape)
            print(f"torch.Size([B,  T,  D])")
        
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    
    def to(self, *args, **kwargs):
        # 기존 `to` 메서드 호출
        super_result = super().to(*args, **kwargs)
        
        # 추가 기능
        if "cuda" in str(args[0]):  # GPU로 이동하는 경우 추가 작업 수행
            self.htm.to("cuda")
            self.bsm.to("cuda")

        return super_result


class MotionMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_temporal,
        d_state=16,
        d_conv=4,
        expand=2,
        nhead = 4,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5, 
        rms_norm: bool = True, 
        residual_in_fp32: bool = True,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        num_layer=1, # value k
        if_divide_out=False,
        init_layer_scale=None,
        aggregate='concat_linear', # 'sum', 'concat_linear'
        log = False,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_temporal = d_temporal
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.rmsnorm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.conv_bias = conv_bias
        self.bias = bias
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.num_layer = num_layer
        self.aggregate = aggregate
        self.if_divide_out = if_divide_out
        self.init_layer_scale = init_layer_scale
        
        self.norms = nn.ModuleList(
            [(nn.LayerNorm if not rms_norm else RMSNorm)(self.d_model, eps=norm_epsilon, **factory_kwargs) for _ in range(self.num_layer)]
        )

        self.encs = nn.ModuleList(
            [MotionMambaBlock(
                d_model,
                d_temporal,
                d_state,
                d_conv,
                expand,
                dt_rank,
                dt_min,
                dt_max,
                dt_init,
                dt_scale,
                dt_init_floor,
                drop_rate,
                drop_path_rate,
                norm_epsilon,
                rms_norm,
                residual_in_fp32,
                conv_bias,
                bias,
                use_fast_path,
                layer_idx,
                device,
                dtype,
                2*k-1,
                if_divide_out,
                init_layer_scale,
                aggregate,
                log,
                **kwargs

            ) for k in range(self.num_layer, 0, -1)]
        )
        
        self.mixer = nn.MultiheadAttention(self.d_model, nhead, dropout=drop_path_rate)
        
        self.decs = nn.ModuleList(
            [MotionMambaBlock(
                d_model,
                d_temporal,
                d_state,
                d_conv,
                expand,
                dt_rank,
                dt_min,
                dt_max,
                dt_init,
                dt_scale,
                dt_init_floor,
                drop_rate,
                drop_path_rate,
                norm_epsilon,
                rms_norm,
                residual_in_fp32,
                conv_bias,
                bias,
                use_fast_path,
                layer_idx,
                device,
                dtype,
                2*k-1,
                if_divide_out,
                init_layer_scale,
                aggregate,
                log,
                **kwargs

            ) for k in range(1, self.num_layer+1)]
        )

    def forward(self, input, context=None, inference_params=None):
        """

        hidden_states: (B, T, D)
        Returns: same shape as hidden_states
        """
                
        batch, seqlen, dim = input.shape
        residuals = deque()

        hidden_state = input

        # Run Motion Mamba Encoder Blocks
        for enc in self.encs:
            hidden_state = enc(hidden_state)
            residuals.append(hidden_state)

        # Run Transformer Mixer Block
        if context:
            hidden_state = self.mixer(query = hidden_state, key = context, value = context)[0]
        else:
            # self-attention
            out = self.mixer(query = hidden_state, key = hidden_state, value = hidden_state)[0]

        # Run Motion Mamba Decoder Blocks
        for dec in self.decs:
            out = out + residuals.pop()
            out = dec(out)
        
        assert not residuals, f"Residuals is not empty: {list(residuals)}"

        return out


    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
    
    def to(self, *args, **kwargs):
        # 기존 `to` 메서드 호출
        super_result = super().to(*args, **kwargs)
        
        # 추가 기능
        if "cuda" in str(args[0]):  # GPU로 이동하는 경우 추가 작업 수행
            for k in range(self.num_layer):
                self.encs[k].to("cuda")
                self.decs[k].to("cuda")

        return super_result
    
