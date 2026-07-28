# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
from palimpsa.ops.palimpsa import chunk_palimpsa, fused_recurrent_palimpsa

# Import Fast Palimpsa
from palimpsa.ops.fast_palimpsa.chunk_fast_palimpsa import chunk_fast_palimpsa


import wandb

if TYPE_CHECKING:
    from fla.models.utils import Cache
    from transformers.processing_utils import Unpack

# =============================================================================
# Distributed helpers
# =============================================================================

def is_master() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class Palimpsa(nn.Module):
    
    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        expand_k: float = 1,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: Optional[int] = None,
        beta_step_rank: int = 128,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        use_residual: bool = True,
        init_diagnosis: bool = True,
        eval_diagnosis: bool = False,
        token_mixer_type: str = 'palimpsa',
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_residual    = use_residual
        self.init_diagnosis  = init_diagnosis
        self.eval_diagnosis  = eval_diagnosis
        self.token_mixer_type = token_mixer_type

        if token_mixer_type == "palimpsa":
            warnings.warn("⚠️ Palimpsa: full metaplasticity (vector β).")
            self.metaplasticity = True
            self.beta_vector    = True
        elif token_mixer_type == "simple_gla":
            warnings.warn("⚠️ Palimpsa: simple_gla (no metaplasticity).")
            self.metaplasticity = False
            self.beta_vector    = False
        elif token_mixer_type == "fast_palimpsa":
            warnings.warn("⚠️ Palimpsa: fast_palimpsa (chunked isotropic approximation). Overriding standard MoPa.")
            self.metaplasticity = True
            self.beta_vector    = True


        self.mode             = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size      = hidden_size
        self.expand_v         = expand_v
        self.expand_k         = expand_k

        self.use_gate       = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size      = conv_size
        self.conv_bias      = conv_bias

        self.head_dim       = head_dim
        self.num_heads      = num_heads
        self.num_v_heads    = num_v_heads if num_v_heads is not None else num_heads
        self.beta_step_rank = beta_step_rank

        self.head_k_dim = int(self.head_dim * self.expand_k)
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim    = int(self.num_heads   * self.head_k_dim)
        self.value_dim  = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx  = layer_idx


        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError("Invalid value_dim configuration.")

        # --- Projections ---
        self.q_proj = nn.Linear(hidden_size, self.key_dim,   bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim,   bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.b_rank_proj = nn.Linear(hidden_size,         self.beta_step_rank, bias=False)
        self.b_proj      = nn.Linear(self.beta_step_rank, self.value_dim,      bias=False)
        if not self.beta_vector:
            for p in self.b_rank_proj.parameters():
                p.requires_grad = False
            for p in self.b_proj.parameters():
                p.requires_grad = False

        self.b_scale            = nn.Parameter(torch.ones(self.num_v_heads), requires_grad=self.metaplasticity)
        self.b_scale._no_weight_decay = True

        self.bs_proj            = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.Ip_log             = nn.Parameter(torch.zeros(self.num_v_heads), requires_grad=False)
        self.Ip_log._no_weight_decay = True

        self.dt_proj            = nn.Linear(hidden_size, self.num_heads, bias=False)
        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log              = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias            = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim,   kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim,   kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')

        self.D = nn.Parameter(torch.ones(self.num_v_heads))
        self.D._no_weight_decay = True

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
s

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _diag_init(self, k, b, b_scale, dt, A):
        if not (wandb.run is not None and is_master()):
            return
        with torch.no_grad():
            decay = torch.exp(-A * dt)
            n_val = 1.0 / (1.0 - decay + 1e-6)
            n_avg = n_val.mean(dim=(0, 1))

            metrics = {
                f"diag_init/L{self.layer_idx}_b_rank_proj_std": self.b_rank_proj.weight.std().item(),
                f"diag_init/L{self.layer_idx}_b_proj_std":      self.b_proj.weight.std().item(),
                f"diag_init/L{self.layer_idx}_k_proj_std":      self.k_proj.weight.std().item(),
                f"diag_init/L{self.layer_idx}_b_scale":         b_scale.mean().item(),
            }
            if b is not None:
                metrics[f"diag_init/L{self.layer_idx}_b_output_std"] = b.std().item()
            for h in range(len(n_avg)):
                metrics[f"diag_init/L{self.layer_idx}_N_avg/H{h}"] = n_avg[h].item()
            wandb.log(metrics, commit=False)

    def _diag_eval(self, final_I, b, dt, A):
        if final_I is None or not (wandb.run is not None and is_master()):
            return
        with torch.no_grad():
            metrics = {}
            H = final_I.shape[1]
            current_b_scales = F.softplus(self.b_scale).detach()
            decay = torch.exp(-A * dt)
            n_val = 1.0 / (1.0 - decay + 1e-6)
            n_avg = n_val.mean(dim=(0, 1))
            for h in range(H):
                state_h = final_I[:, h, ...]
                metrics[f"diag_eval/L{self.layer_idx}_I_Range/H{h}"] = (state_h.max() - state_h.min()).item()
                metrics[f"diag_eval/L{self.layer_idx}_I_Mean/H{h}"]  = state_h.mean().item()
                metrics[f"diag_eval/L{self.layer_idx}_I_Std/H{h}"]   = state_h.std().item()
                if b is not None:
                    metrics[f"diag_eval/L{self.layer_idx}_b_std/H{h}"] = b[:, :, h, :].std().item()
                metrics[f"diag_eval/L{self.layer_idx}_b_scale/H{h}"] = current_b_scales[h].item()
                metrics[f"diag_eval/L{self.layer_idx}_N_avg/H{h}"]   = n_avg[h].item()
                metrics[f"diag_eval/L{self.layer_idx}_A/H{h}"]       = A[h].item()
                metrics[f"diag_eval/L{self.layer_idx}_dt_avg/H{h}"]  = dt[:, :, h].mean().item()
            wandb.log(metrics, commit=False)

    # ------------------------------------------------------------------
    # Token-mixer dispatch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_recurrent_state(recurrent_state):
        if recurrent_state is None:
            return None, None
        if isinstance(recurrent_state, (list, tuple)):
            a = recurrent_state[0] if len(recurrent_state) > 0 else None
            b = recurrent_state[1] if len(recurrent_state) > 1 else None
            return a, b
        return recurrent_state, None


    def _run_fast_palimpsa(self, q, k, v, b, dt, A, Ip, recurrent_state, use_cache, cu_seqlens, mode):
        if cu_seqlens is not None:
            warnings.warn("fast_palimpsa does not support cu_seqlens yet. Things might break.")
        if recurrent_state is not None:
            warnings.warn("fast_palimpsa does not support recurrent_state yet. Dropping state!")

        o = chunk_fast_palimpsa(q=q, k=k, v=v, b=b, gt=dt, g=A, Ip=Ip, chunk_size=16)

        if use_cache:
            warnings.warn("fast_palimpsa does not return a final state. Cache will be None!")
            return o, None
        return o, None

    
    def _run_simple_gla(self, q, k, v, g_log, recurrent_state, use_cache, cu_seqlens, mode):
        initial_state = recurrent_state if recurrent_state is not None else None
        kernel = chunk_simple_gla if mode == 'chunk' else fused_recurrent_simple_gla
        outputs = kernel(q=q, k=k, v=v, g=g_log,
                         initial_state=initial_state,
                         output_final_state=use_cache,
                         cu_seqlens=cu_seqlens)
        o, final_state = outputs if isinstance(outputs, tuple) else (outputs, None)
        new_state = final_state if use_cache else None
        return o, new_state

    def _run_palimpsa(self, q, k, v, b, dt, A, Ip, recurrent_state, use_cache, cu_seqlens, mode):
        active_mu, active_I = self._unpack_recurrent_state(recurrent_state)
        if mode == 'chunk':
            outputs = chunk_palimpsa(q=q, k=k, v=v, b=b, gt=dt, g=A, Ip=Ip,
                                     output_final_state=use_cache,
                                     cu_seqlens=cu_seqlens, chunk_size=16,
                                     initial_mu_state=active_mu,
                                     initial_I_state=active_I)
        else:
            outputs = fused_recurrent_palimpsa(q=q, k=k, v=v, b=b, gt=dt, g=A, Ip=Ip,
                                               initial_mu_state=active_mu,
                                               initial_I_state=active_I,
                                               output_final_state=use_cache,
                                               cu_seqlens=cu_seqlens)
        if use_cache:
            o, final_mu, final_I = outputs
            new_state = (final_mu, final_I)
        else:
            o = outputs
            new_state = None
        return o, new_state

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        batch_size, q_len, _ = hidden_states.shape

        mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')

        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."),
                indices,
            ).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None and last_state.get('conv_state') is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states),
                                            cache=conv_state_q,
                                            output_final_state=use_cache,
                                            cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states),
                                            cache=conv_state_k,
                                            output_final_state=use_cache,
                                            cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states),
                                            cache=conv_state_v,
                                            output_final_state=use_cache,
                                            cu_seqlens=cu_seqlens)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        x = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q = repeat(q, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads)
            k = repeat(k, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads)

        dt = F.softplus(self.dt_proj(hidden_states).float() + self.dt_bias)
        A  = self.A_log.float().exp()
        q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
        bs   = torch.sigmoid(self.bs_proj(hidden_states).float()).to(hidden_states.dtype)
        v = x * bs.unsqueeze(-1)

        if self.beta_vector:
            b_raw = self.b_proj(self.b_rank_proj(hidden_states)).float()
            b_raw = rearrange(b_raw, '... (h d) -> ... h d', d=self.head_v_dim)
            b = torch.sigmoid(b_raw) * F.softplus(self.b_scale.view(1, 1, -1, 1).float())
            b = (b * bs.unsqueeze(-1)).to(hidden_states.dtype)
        else:
            b = None

        Ip = torch.exp(self.Ip_log.float())

        if self.init_diagnosis and self.training and not hasattr(self, "_mangled") and self.layer_idx == 0:
            self._diag_init(k, b, self.b_scale, dt, A)
            self._mangled = True

        recurrent_state = None
        if last_state is not None:
            recurrent_state = (last_state.get('recurrent_state')
                                if isinstance(last_state, dict) else last_state[0])

        # ------------------------------------------------------------------
        # Token-mixer dispatch
        # ------------------------------------------------------------------
        g_log = -dt * A
        n_states_used = None
        mopa_states_used = None

        
        if self.token_mixer_type == "simple_gla":
            o, recurrent_state = self._run_simple_gla(
                q, k, v, g_log, recurrent_state, use_cache, cu_seqlens, mode,
            )

        elif self.token_mixer_type == "palimpsa":
            o, recurrent_state = self._run_palimpsa(
                q, k, v, b, dt, A, Ip, recurrent_state, use_cache, cu_seqlens, mode,
            )

        elif self.token_mixer_type == "fast_palimpsa":
            o, recurrent_state = self._run_fast_palimpsa(
                q, k, v, b, dt, A, Ip, recurrent_state, use_cache, cu_seqlens, mode,
            )

        else:
            raise ValueError(f"Invalid token_mixer_type: {self.token_mixer_type}")

        # ------------------------------------------------------------------
        # Past-state writeback
        # ------------------------------------------------------------------
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=((conv_state_q, conv_state_k, conv_state_v)
                            if self.use_short_conv else None),
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        # ------------------------------------------------------------------
        # Output head
        # ------------------------------------------------------------------
        if self.use_residual:
            o = o + x * self.D[None, None, :, None]

        if self.use_gate:
            g_proj = rearrange(self.g_proj(hidden_states),
                                '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g_proj)
        else:
            o = self.o_norm(o)
        o = self.o_proj(rearrange(o, 'b t h d -> b t (h d)'))

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values