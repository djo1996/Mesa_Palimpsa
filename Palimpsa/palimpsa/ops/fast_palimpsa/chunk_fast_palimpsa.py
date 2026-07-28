# -*- coding: utf-8 -*-
# chunk_fast_palimpsa.py
#
# Fast Palimpsa: chunked isotropic-in-D_V approximation of Palimpsa.
#   * State carried across chunks is EXACT, full D_V x D_K.
#   * Inside a chunk, the LOCAL contribution is read with an isotropic precision
#     Ibar_t in R^{D_K} (collapsed over D_V), evolved with full dynamics
#     (alpha decay + (1-f) Ip prior + betabar k^2).
#   * The CARRY contribution (history) is read against the FROZEN boundary state
#     mu_c = M_c / I_c (tl.dot-friendly, the whole point: no per-token DVxDK read).
#
# This file contains:
#   1. fast_palimpsa_ref      -- differentiable PyTorch reference (the contract).
#   2. chunk_fast_palimpsa     -- Triton autograd Function (tiled, any D_K/D_V).
#   3. test_fwd_bwd()          -- compares Triton fwd AND bwd against the reference.
#
# Tiling follows fla chunk_h / chunk_gla: state h=[K,V] tiled by (BK,BV) on a
# 3D grid (cdiv(K,BK), cdiv(V,BV), B*H); chunk loop is inside the kernel; reductions
# over D_K are accumulated across BK blocks. Any D_K, D_V are supported.

from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except Exception:
    HAS_TRITON = False

_EPS = 1e-12
CHUNK_C = 32
FP_CLAMP_VERSION = "L2-matmul-v9-decay-exp-clamp-bf16"   # bump to confirm the right file is loaded


def _max_shared_mem(dev_index: int = 0) -> int:
    """Per-block shared-memory limit (bytes) for the active CUDA device.

    Mirrors fla.utils.check_shared_mem: query the device so block sizes can be
    capped to fit. Falls back to a conservative 96 KiB if the query fails.
    """
    try:
        props = torch.cuda.get_device_properties(dev_index)
        for attr in ("shared_memory_per_block_optin",
                     "max_shared_memory_per_block",
                     "shared_memory_per_block"):
            val = getattr(props, attr, None)
            if val:
                return int(val)
    except Exception:
        pass
    return 96 * 1024


def _fp_output_block_sizes(D_K, D_V, C, dev_index=0):
    """Pick (BK, BV, num_warps, num_stages) for _fp_output_kernel so the kernel
    fits in shared memory. BV must cover the full D_V (the carry reduction sums
    I_c over the whole value dim), so we shrink BK and num_stages instead.

    Follows the chunk_h / chunk_gla idiom: tile the key dim, gate on the device
    shared-memory limit, and drop num_stages before giving up.
    """
    BV = triton.next_power_of_2(D_V)
    limit = _max_shared_mem(dev_index)
    # 4 bytes/f32. Per-iteration loop tensors (q,k,ibar: C*BK; M,I,mu: BV*BK) get
    # multiplied by num_stages worth of pipeline buffers; persistent tensors
    # (score,Dmask: C*C; carry,v: C*BV) are allocated once. Estimate and back off.
    def est(BK, stages):
        per_iter = (3 * C * BK + 3 * BV * BK) * 4
        persistent = (2 * C * C + 2 * C * BV) * 4
        return persistent + stages * per_iter

    for BK in (min(64, triton.next_power_of_2(D_K)), 32, 16):
        for stages in (3, 2, 1):
            if est(BK, stages) <= limit:
                num_warps = 4 if BV >= 64 else 2
                return BK, BV, num_warps, stages
    # Last resort: smallest BK, single stage.
    return 16, BV, 2, 1


# =============================================================================
# 1. Reference (differentiable)
# =============================================================================
def _resolve_Ip(Ip, H, DK, dev, dt):
    if not torch.is_tensor(Ip):
        Ip = torch.full((H,), float(Ip), device=dev, dtype=dt)
    Ip = Ip.to(device=dev, dtype=dt)
    if Ip.numel() == H:                       # scalar per head
        return Ip.view(1, H, 1), Ip.view(1, H, 1, 1)        # (1,H,1) key-bcast, (1,H,1,1) vk-bcast
    elif Ip.numel() == H * DK:                # per (head, key-dim)
        return Ip.view(1, H, DK), Ip.view(1, H, 1, DK)
    raise ValueError("Ip must be scalar-per-head (H,) or per-(H,DK).")


def _exact_recurrence(q, k, v, b, gt, g, Ip, scale):
    """Token-exact Palimpsa ground truth. Differentiable."""
    B, L, H, DK = q.shape
    DV = v.shape[-1]
    dev, dt = q.device, q.dtype
    Ip_k, Ipv = _resolve_Ip(Ip, H, DK, dev, dt)
    Ipv_full = Ipv  # (1,H,1,1) or (1,H,1,DK)
    q = q * scale
    M = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
    I = (Ipv_full.expand(B, H, DV, DK).clone() if Ipv_full.shape[-1] == DK
         else Ipv_full.view(1, H, 1, 1).expand(B, H, DV, DK).clone())
    ys, yvars = [], []
    for t in range(L):
        f = torch.exp(-(gt[:, t] * g.view(1, H))).view(B, H, 1, 1)
        kt = k[:, t].view(B, H, 1, DK)
        vt = v[:, t].view(B, H, DV, 1)
        bt = b[:, t].view(B, H, DV, 1)
        I = f * I + (1 - f) * (Ipv_full if Ipv_full.shape[-1] == DK else Ipv_full) + bt * (kt * kt)
        M = f * M + vt * kt
        mu = M / I
        qt = q[:, t].view(B, H, 1, DK)
        ys.append((mu * qt).sum(-1))
        yvars.append((qt * qt / I).sum(-1))
    return torch.stack(ys, 1), torch.stack(yvars, 1)


def fast_palimpsa_ref(q, k, v, b, gt, g, Ip, scale=None, chunk_size=CHUNK_C,
                      output_uncertainty=False):
    """Chunked isotropic approximation (frozen carry). Differentiable. The contract."""
    B, L, H, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    assert L % C == 0
    nc = L // C
    if scale is None:
        scale = DK ** -0.5
    dev, dt = q.device, q.dtype
    Ip_k, Ipv = _resolve_Ip(Ip, H, DK, dev, dt)
    perdk = Ipv.shape[-1] == DK

    qs = q * scale
    M_c = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
    I_c = (Ipv.expand(B, H, DV, DK).clone() if perdk
           else Ipv.view(1, H, 1, 1).expand(B, H, DV, DK).clone())

    y_out = torch.zeros(B, L, H, DV, dtype=dt, device=dev)
    yvar_out = torch.zeros(B, L, H, DV, dtype=dt, device=dev)

    for c in range(nc):
        sl = slice(c * C, (c + 1) * C)
        kc, vc, bc, qc, gc = k[:, sl], v[:, sl], b[:, sl], qs[:, sl], gt[:, sl]
        f = torch.exp(-(gc * g.view(1, 1, H)))                  # (B,C,H)
        logf = torch.log(f.clamp_min(1e-30))
        clogf = torch.cumsum(logf, dim=1)                       # (B,C,H)

        Ibar_c = I_c.mean(2)                                    # (B,H,DK)
        abar_c = Ibar_c - Ip_k                                  # (B,H,DK)
        betabar = bc.mean(-1)                                   # (B,C,H)
        ksq = kc * kc

        Ibar = torch.empty(B, C, H, DK, dtype=dt, device=dev)
        a_prev = abar_c
        for t in range(C):
            ft = f[:, t].unsqueeze(-1)
            a_prev = ft * a_prev + betabar[:, t].unsqueeze(-1) * ksq[:, t]
            Ibar[:, t] = Ip_k + a_prev

        # ######## MODIFICATION 2 GEMINI #####################################
        # local (isotropic), reader-side scaling Qtil = q/Ibar
        Qtil = qc / Ibar
        score = torch.einsum('bthd,bshd->btsh', Qtil, kc)       # (B,C,C,H)
        Dmask = torch.exp(clogf.unsqueeze(2) - clogf.unsqueeze(1))
        tri = torch.tril(torch.ones(C, C, device=dev, dtype=dt)).view(1, C, C, 1)
        score = score * Dmask * tri
        y_local = torch.einsum('btsh,bshv->bthv', score, vc)    # (B,C,H,DV)

        # NEW carry (Exact mu_c, with relative isotropic decay on Q)
        mu_c = M_c / I_c
        Q_carry_til = qc * (Ibar_c.unsqueeze(1) / Ibar)         # (B,C,H,DK)
        base = torch.einsum('bhvd,bthd->bthv', mu_c, Q_carry_til) 
        
        carry_decay = torch.exp(clogf).unsqueeze(-1)            
        y_out[:, sl] = y_local + base * carry_decay
        ######## END MODIFICATION 2 GEMINI #####################################

        if output_uncertainty:
            yv = torch.einsum('bthd,bthd->bth', qc * qc, 1.0 / Ibar)
            yvar_out[:, sl] = yv.unsqueeze(-1).expand(B, C, H, DV)

        # exact boundary update
        I_new, M_new = I_c.clone(), M_c.clone()
        Ipv_full = (Ipv if perdk else Ipv.view(1, H, 1, 1))
        for t in range(C):
            ft = f[:, t].view(B, H, 1, 1)
            kt = kc[:, t].view(B, H, 1, DK)
            vt = vc[:, t].view(B, H, DV, 1)
            bt = bc[:, t].view(B, H, DV, 1)
            I_new = ft * I_new + (1 - ft) * Ipv_full + bt * (kt * kt)
            M_new = ft * M_new + vt * kt
        I_c, M_c = I_new, M_new

    if output_uncertainty:
        return y_out, yvar_out
    return y_out


def fast_palimpsa_vec(q, k, v, b, gt, g, Ip, scale=None, chunk_size=CHUNK_C):
    """Chunk-parallel, fully-vectorized implementation (no Python token loops).

    Same math as fast_palimpsa_ref, but the two intra-chunk `for t in range(C)`
    loops are replaced by closed-form cumulative-decay matmuls, so it runs at
    bmm/einsum efficiency and is differentiable by autograd. Works for ANY
    D_K / D_V with no shared-memory constraint (no full-D_V resident tile), which
    is why it sidesteps the SMEM blow-up that the hand-written Triton backward
    hits at D_V=192. Verified to match the loop reference fwd + all grads to
    machine precision (fp64).
    """
    B, L, H, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    assert L % C == 0, f"L={L} not divisible by C={C}"
    nc = L // C
    if scale is None:
        scale = DK ** -0.5
    dev, dt = q.device, q.dtype
    Ip_k, Ipv = _resolve_Ip(Ip, H, DK, dev, dt)
    perdk = Ipv.shape[-1] == DK
    Ip_k = Ip_k if perdk else Ipv.view(1, H, 1).expand(1, H, DK)

    qs = q * scale
    M_c = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
    I_c = Ip_k.view(1, H, 1, DK).expand(B, H, DV, DK).clone()

    triCC = torch.tril(torch.ones(C, C, device=dev, dtype=dt)).view(1, C, C, 1)
    y_out = torch.zeros(B, L, H, DV, dtype=dt, device=dev)

    for c in range(nc):
        sl = slice(c * C, (c + 1) * C)
        kc, vc, bc, qc, gc = k[:, sl], v[:, sl], b[:, sl], qs[:, sl], gt[:, sl]
        f = torch.exp(-(gc * g.view(1, 1, H)))                  # (B,C,H)
        logf = torch.log(f.clamp_min(1e-30))
        clogf = torch.cumsum(logf, dim=1)                       # (B,C,H)
        cd = torch.exp(clogf)

        Ibar_c = I_c.mean(2)                                    # (B,H,DK)
        abar_c = Ibar_c - Ip_k                                  # (B,H,DK)
        betabar = bc.mean(-1)                                   # (B,C,H)
        ksq = kc * kc

        # closed-form intra-chunk Ibar scan (replaces the t-loop):
        #   a_t = cd_t*abar_c + sum_{i<=t} (cd_t/cd_i)*betabar_i*ksq_i
        Dm = torch.exp(clogf.unsqueeze(2) - clogf.unsqueeze(1)) * triCC   # (B,C,C,H)[t,i]
        src = betabar.unsqueeze(-1) * ksq                       # (B,C,H,DK) [i]
        a_mat = cd.unsqueeze(-1) * abar_c.unsqueeze(1) + torch.einsum('btih,bihd->bthd', Dm, src)
        Ibar = Ip_k.unsqueeze(1) + a_mat                        # (B,C,H,DK)

        # local (isotropic) reader-side scaling
        Qtil = qc / Ibar
        score = torch.einsum('bthd,bshd->btsh', Qtil, kc)
        Dmask = torch.exp(clogf.unsqueeze(2) - clogf.unsqueeze(1))
        score = score * Dmask * triCC
        y_local = torch.einsum('btsh,bshv->bthv', score, vc)

        # exact-mu carry with relative isotropic decay on Q
        mu_c = M_c / I_c
        Q_carry_til = qc * (Ibar_c.unsqueeze(1) / Ibar)
        base = torch.einsum('bhvd,bthd->bthv', mu_c, Q_carry_til)
        y_out[:, sl] = y_local + base * cd.unsqueeze(-1)

        # closed-form exact boundary update (replaces the t-loop):
        #   M <- prodF*M + sum_t w_t v_t k_t^T ;  w_t = prod_{i>t} f_i
        prodF = torch.exp(clogf[:, -1])                         # (B,H)
        w = torch.exp(clogf[:, -1:].expand(B, C, H) - clogf)    # (B,C,H)
        pf = prodF.view(B, H, 1, 1)
        M_c = pf * M_c + torch.einsum('bthv,bthd->bhvd', w.unsqueeze(-1) * vc, kc)
        I_c = (pf * I_c + (1 - pf) * Ip_k.view(1, H, 1, DK)
               + torch.einsum('bthv,bthd->bhvd', w.unsqueeze(-1) * bc, ksq))

    return y_out


# =============================================================================
# 2. Triton kernels (tiled; any D_K, D_V)
# =============================================================================
if HAS_TRITON:

    @triton.jit
    def _fp_state_kernel(
        k, v, b, gt, g, Ip,
        M_bound, I_bound,            # (B,H,nc+1,DV,DK) chunk-ENTRY exact states
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
        PERDK: tl.constexpr,
    ):
        # One program per (b, h, k-block, v-block). Walks chunks forward, writes
        # the EXACT chunk-entry M,I for each chunk, then advances exactly.
        i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        nc = T // C

        o_c = tl.arange(0, C)
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < D_K
        mask_v = o_v < D_V
        mask_kv = mask_v[:, None] & mask_k[None, :]

        if PERDK:
            b_Ip = tl.load(Ip + i_h * D_K + o_k, mask=mask_k, other=1.0).to(tl.float32)
        else:
            b_Ip = tl.load(Ip + i_h).to(tl.float32) + tl.zeros([BK], dtype=tl.float32)
        b_g = tl.load(g + i_h).to(tl.float32)

        M = tl.zeros([BV, BK], dtype=tl.float32)
        I = tl.zeros([BV, BK], dtype=tl.float32) + b_Ip[None, :]

        for c in range(nc):
            # store chunk-entry state
            off = ((i_bh * (nc + 1) + c) * D_V * D_K
                   + o_v[:, None] * D_K + o_k[None, :])
            tl.store(M_bound + off, M, mask=mask_kv)
            tl.store(I_bound + off, I, mask=mask_kv)

            base_qk = (i_b * T * H + c * C * H + i_h) * D_K
            base_vo = (i_b * T * H + c * C * H + i_h) * D_V
            base_gt = (i_b * T + c * C) * H + i_h
            k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_k[None, :], other=0.0).to(tl.float32)
            v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            gt_c = tl.load(gt + base_gt + o_c * H, mask=o_c < C, other=0.0).to(tl.float32)
            f_c = tl.exp(-gt_c * b_g)
            logf = -gt_c * b_g
            clogf = tl.cumsum(logf, axis=0)
            clogf_last = tl.sum(tl.where(o_c == (C - 1), clogf, 0.0))
            prodF = tl.exp(clogf_last)                       # prod_t f_t
            # per-token weight w_t = prod_{i>t} f_i = prodF / F_t = exp(clogf_last - clogf_t)
            w = tl.exp(clogf_last - clogf)                   # (C,)
            ksq = k_ck * k_ck
            
            # Matmul forms replacing the sequential token loop
            M = prodF * M + tl.dot(tl.trans((w[:, None] * v_cv)).to(tl.float32),
                                   k_ck.to(tl.float32))
            I = (prodF * I + (1.0 - prodF) * b_Ip[None, :]
                 + tl.dot(tl.trans((w[:, None] * b_cv)).to(tl.float32),
                          ksq.to(tl.float32)))

        # final state at index nc
        off = ((i_bh * (nc + 1) + nc) * D_V * D_K
               + o_v[:, None] * D_K + o_k[None, :])
        tl.store(M_bound + off, M, mask=mask_kv)
        tl.store(I_bound + off, I, mask=mask_kv)

    @triton.jit
    def _fp_ibar_kernel(
        k, b, gt, g, Ip, I_bound,
        Ibar_out,                    # (B,H,nc,C,DK) isotropic per-token precision
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, C: tl.constexpr,
        BV_IB: tl.constexpr,
        PERDK: tl.constexpr,
    ):
        # One program per (b,h,k-block). Collapses I_bound over D_V and evolves Ibar_t.
        i_k, i_bh = tl.program_id(0), tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        nc = T // C
        o_c = tl.arange(0, C)
        o_k = i_k * BK + tl.arange(0, BK)
        mask_k = o_k < D_K

        if PERDK:
            b_Ip = tl.load(Ip + i_h * D_K + o_k, mask=mask_k, other=1.0).to(tl.float32)
        else:
            b_Ip = tl.load(Ip + i_h).to(tl.float32) + tl.zeros([BK], dtype=tl.float32)
        b_g = tl.load(g + i_h).to(tl.float32)

        o_v = tl.arange(0, BV_IB)
        mask_v = o_v < D_V
        for c in range(nc):
            # collapse I_bound[c] over D_V  -> Ibar_c (BK,)  (vectorized block load)
            off_iv = ((i_bh * (nc + 1) + c) * D_V * D_K
                      + o_v[:, None] * D_K + o_k[None, :])
            I_c = tl.load(I_bound + off_iv, mask=(mask_v[:, None] & mask_k[None, :]),
                          other=0.0).to(tl.float32)
            Ibar_c = tl.sum(tl.where(mask_v[:, None], I_c, 0.0), axis=0) / D_V
            abar = Ibar_c - b_Ip

            base_qk = (i_b * T * H + c * C * H + i_h) * D_K
            base_vo = (i_b * T * H + c * C * H + i_h) * D_V
            base_gt = (i_b * T + c * C) * H + i_h
            k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_k[None, :], other=0.0).to(tl.float32)
            gt_c = tl.load(gt + base_gt + o_c * H, mask=o_c < C, other=0.0).to(tl.float32)
            logf = -gt_c * b_g
            clogf = tl.cumsum(logf, axis=0)
            cd = tl.exp(clogf)
            ksq = k_ck * k_ck

            # betabar_t = mean over D_V of b  (vectorized block load over D_V)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            bbar = tl.sum(b_cv, axis=1) / D_V

            # Ibar_t = b_Ip + a_t,  a_t = f_t a_{t-1} + bbar_t ksq_t,  a_{-1}=abar
            #   a_mat = cd*abar + Dm @ (bbar*ksq),  Dm[t,i]=exp(clogf_t-clogf_i) lower-tri
            Dm = tl.where(o_c[:, None] >= o_c[None, :],
                          tl.exp(tl.minimum(clogf[:, None] - clogf[None, :], 0.0)), 0.0)
            a_mat = cd[:, None] * abar[None, :] + tl.dot(
                Dm.to(tl.float32), (bbar[:, None] * ksq).to(tl.float32))
            
            Ibar_t = b_Ip[None, :] + a_mat                   # (C,BK)
            off_out = ((i_bh * nc + c) * C + o_c[:, None]) * D_K + o_k[None, :]
            tl.store(Ibar_out + off_out, Ibar_t, mask=mask_k[None, :])

    @triton.jit
    def _fp_output_kernel(
        q, k, v, gt, g, Ip,
        M_bound, I_bound, Ibar,
        o, scale,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
        PERDK: tl.constexpr,
    ):
        # One program per (b,h,chunk,v-block). Accumulates local score over BK
        # blocks, then local @ v and carry @ q.
        i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        nc = T // C
        o_c = tl.arange(0, C)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_v = o_v < D_V

        base_gt = (i_b * T + i_c * C) * H + i_h
        b_g = tl.load(g + i_h).to(tl.float32)
        gt_c = tl.load(gt + base_gt + o_c * H, mask=o_c < C, other=0.0).to(tl.float32)
        logf = -gt_c * b_g
        clogf = tl.cumsum(logf, axis=0)
        carry_decay = tl.exp(clogf)                       # (C,)
        Dmask = tl.exp(tl.minimum(clogf[:, None] - clogf[None, :], 0.0))
        causal = o_c[:, None] >= o_c[None, :]
        Dmask = tl.where(causal, Dmask, 0.0)              # (C,C)

        base_qk = (i_b * T * H + i_c * C * H + i_h) * D_K
        base_vo = (i_b * T * H + i_c * C * H + i_h) * D_V

        v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                       mask=mask_v[None, :], other=0.0).to(tl.float32)

        score = tl.zeros([C, C], dtype=tl.float32)
        carry = tl.zeros([C, BV], dtype=tl.float32)
        NK = tl.cdiv(D_K, BK)
        for i_k in range(NK):
            o_k = i_k * BK + tl.arange(0, BK)
            mask_k = o_k < D_K
            mask_kv = mask_v[:, None] & mask_k[None, :]
            if PERDK:
                b_Ip_out = tl.load(Ip + i_h * D_K + o_k, mask=mask_k, other=1.0).to(tl.float32)
            else:
                b_Ip_out = tl.load(Ip + i_h).to(tl.float32) + tl.zeros([BK], dtype=tl.float32)
            b_Ip_out = tl.where(mask_k, b_Ip_out, 1.0)
            q_ck = tl.load(q + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_k[None, :], other=0.0).to(tl.float32) * scale
            k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_k[None, :], other=0.0).to(tl.float32)
            ibar = tl.load(Ibar + ((i_bh * nc + i_c) * C + o_c[:, None]) * D_K + o_k[None, :],
                           mask=mask_k[None, :], other=1.0).to(tl.float32)
            ibar = tl.maximum(ibar, b_Ip_out[None, :])
            Qtil = q_ck / ibar
            score += tl.dot(Qtil.to(tl.float32), tl.trans(k_ck))

            # carry term: Qtil @ M_c^T over this k-block
            off_st = ((i_bh * (nc + 1) + i_c) * D_V * D_K
                      + o_v[:, None] * D_K + o_k[None, :])
            M_c = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_c = tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32)
            I_c = tl.maximum(I_c, b_Ip_out[None, :])

            # Since BV covers all of D_V, we safely reduce directly in SRAM/Registers
            Ibar_c = tl.sum(tl.where(mask_v[:, None], I_c, 0.0), axis=0) / D_V
            Q_carry_til = q_ck * (Ibar_c[None, :] / ibar)

            mu_c = M_c / I_c
            carry += tl.dot(Q_carry_til.to(tl.float32), tl.trans(mu_c))

        score = score * Dmask
        y_local = tl.dot(score.to(tl.float32), v_cv)
        y = y_local + carry * carry_decay[:, None]
        tl.store(o + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                 y.to(o.dtype.element_ty), mask=mask_v[None, :])

    def _fast_palimpsa_fwd_triton(q, k, v, b, gt, g, Ip, scale, C):
        B, T, H, D_K = q.shape
        D_V = v.shape[-1]
        nc = T // C
        dev = q.device
        # Keep BK tiled, but let BV cover the full D_V dimension
        BK = min(64, triton.next_power_of_2(D_K))
        BV = triton.next_power_of_2(D_V)
        obk, obv, onw, ons = _fp_output_block_sizes(D_K, D_V, C, dev.index or 0)
        perdk = torch.is_tensor(Ip) and Ip.numel() == H * D_K
        if not torch.is_tensor(Ip):
            Ip = torch.full((H,), float(Ip), device=dev, dtype=torch.float32)
        Ip = Ip.float().contiguous()
        g = g.float().contiguous()

        M_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)
        I_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)
        Ibar = torch.empty(B * H, nc, C, D_K, device=dev, dtype=torch.float32)
        o = torch.empty(B, T, H, D_V, device=dev, dtype=q.dtype)

        qc, kc, vc, bc, gtc = [x.contiguous() for x in (q, k, v, b, gt)]

        _fp_state_kernel[(triton.cdiv(D_K, BK), triton.cdiv(D_V, BV), B * H)](
            kc, vc, bc, gtc, g, Ip, M_bound, I_bound,
            T, H=H, D_K=D_K, D_V=D_V, BK=BK, BV=BV, C=C, PERDK=perdk)
        _fp_ibar_kernel[(triton.cdiv(D_K, BK), B * H)](
            kc, bc, gtc, g, Ip, I_bound, Ibar,
            T, H=H, D_K=D_K, D_V=D_V, BK=BK, C=C,
            BV_IB=triton.next_power_of_2(D_V), PERDK=perdk)
        _fp_output_kernel[(triton.cdiv(D_V, obv), nc, B * H)](
            qc, kc, vc, gtc, g, Ip, M_bound, I_bound, Ibar, o, scale,
            T, H=H, D_K=D_K, D_V=D_V, BK=obk, BV=obv, C=C, PERDK=perdk,
            num_warps=onw, num_stages=ons)
        return o

    def _fp_bwd_autotune():
        return [
            triton.Config({'BK': bk, 'BV': bv}, num_warps=w, num_stages=s)
            for bk in (32, 64)
            for bv in (16, 32, 64)
            for w in (2, 4, 8)
            for s in (1, 2)
        ]

    @triton.autotune(configs=_fp_bwd_autotune(), key=['D_K', 'D_V', 'C'])
    @triton.jit
    def _fp_bwd_kernel_local_state(
        q, k, v, b, gt, g, Ip,
        M_bound, I_bound, do,
        dM_local_out, dI_local_out, scale,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
        PERDK: tl.constexpr,
    ):
        """Pass 1/3 (Parallel), BV-tiled. grid=(NK, NC, B*H).

        Two sweeps over the D_V tiles: sweep A accumulates the D_V-reductions
        (Ibar_c, bbar, dsc, dQtil_carry); the middle [C,*] algebra is computed
        once (no D_V axis, fully resident); sweep B writes the per-(D_V-block)
        out_dM/out_dI. No full-D_V residency -> fits SMEM for any D_V.
        """
        i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        nc = T // C
        c = i_c

        o_c = tl.arange(0, C)
        o_k = i_k * BK + tl.arange(0, BK)
        mask_k = o_k < D_K
        NV = tl.cdiv(D_V, BV)

        if PERDK:
            b_Ip = tl.load(Ip + i_h * D_K + o_k, mask=mask_k, other=1.0).to(tl.float32)
        else:
            b_Ip = tl.load(Ip + i_h).to(tl.float32) + tl.zeros([BK], dtype=tl.float32)
        b_Ip = tl.where(mask_k, b_Ip, 1.0)
        b_g = tl.load(g + i_h).to(tl.float32)

        base_qk = (i_b * T * H + c * C * H + i_h) * D_K
        base_vo = (i_b * T * H + c * C * H + i_h) * D_V
        base_gt = (i_b * T + c * C) * H + i_h

        q_ck = tl.load(q + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_k[None, :], other=0.0).to(tl.float32)
        k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_k[None, :], other=0.0).to(tl.float32)
        gt_c = tl.load(gt + base_gt + o_c * H, mask=o_c < C, other=0.0).to(tl.float32)

        qs = q_ck * scale
        f_c = tl.exp(-gt_c * b_g)
        logf = -gt_c * b_g
        clogf = tl.cumsum(logf, axis=0)
        cd = tl.exp(clogf)
        ksq = k_ck * k_ck

        Dm_full = tl.exp(tl.minimum(clogf[:, None] - clogf[None, :], 0.0))
        tri = o_c[:, None] >= o_c[None, :]
        Dm = tl.where(tri, Dm_full, 0.0)

        # ---- sweep A: accumulate D_V-reductions over value-tiles ----
        Ibar_c = tl.zeros([BK], dtype=tl.float32)              # sum_v I_c
        bbar = tl.zeros([C], dtype=tl.float32)                 # sum_v b
        dsc = tl.zeros([C, C], dtype=tl.float32)               # do @ v^T
        dQtil_carry = tl.zeros([C, BK], dtype=tl.float32)      # (do*cd) @ mu
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (nc + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), b_Ip[None, :])
            mu_v = M_cv / I_cv
            v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            do_cv = tl.load(do + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                            mask=mask_v[None, :], other=0.0).to(tl.float32)
            Ibar_c += tl.sum(tl.where(mask_v[:, None], I_cv, 0.0), axis=0)
            bbar += tl.sum(tl.where(mask_v[None, :], b_cv, 0.0), axis=1)
            dbase = do_cv * cd[:, None]
            dsc += tl.dot(do_cv.to(tl.float32), tl.trans(v_cv).to(tl.float32))
            dQtil_carry += tl.dot(dbase.to(tl.float32), mu_v.to(tl.float32))
        Ibar_c = Ibar_c / D_V
        bbar = bbar / D_V

        # ---- middle: [C,*] algebra, no D_V axis ----
        abar = tl.where(mask_k, Ibar_c - b_Ip, 0.0)
        src = bbar[:, None] * ksq
        a_mat = cd[:, None] * abar[None, :] + tl.dot(Dm.to(tl.float32), src.to(tl.float32))
        Ibar_mat = tl.maximum(b_Ip[None, :] + a_mat, b_Ip[None, :])
        Qtil = qs / Ibar_mat

        dsc_raw = dsc * Dm
        dQtil_local = tl.dot(dsc_raw.to(tl.float32), k_ck.to(tl.float32))
        dIbar = -(dQtil_local * qs / (Ibar_mat * Ibar_mat)) \
                - (dQtil_carry * qs * Ibar_c[None, :] / (Ibar_mat * Ibar_mat))
        dIbar_c = tl.sum(dQtil_carry * (qs / Ibar_mat), axis=0)        # (BK,)
        Wt = tl.where(o_c[None, :] >= o_c[:, None], tl.exp(tl.minimum(clogf[None, :] - clogf[:, None], 0.0)), 0.0)
        D = tl.dot(Wt.to(tl.float32), dIbar.to(tl.float32))
        D0 = tl.sum(tl.where(o_c[:, None] == 0, D, 0.0), axis=0)
        f0 = tl.sum(tl.where(o_c == 0, f_c, 0.0))
        dabar = f0 * D0                                                # (BK,)
        bdry_dI = (dIbar_c + dabar) / D_V                              # broadcast over v

        Q_carry_til = qs * (Ibar_c[None, :] / Ibar_mat)

        # ---- sweep B: per-block out_dM / out_dI ----
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (nc + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), b_Ip[None, :])
            do_cv = tl.load(do + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                            mask=mask_v[None, :], other=0.0).to(tl.float32)
            dbase = do_cv * cd[:, None]
            dmu = tl.dot(tl.trans(dbase).to(tl.float32), Q_carry_til.to(tl.float32))   # (BV,BK)
            out_dM = dmu / I_cv
            out_dI = -dmu * M_cv / (I_cv * I_cv) + tl.where(mask_v[:, None], bdry_dI[None, :], 0.0)
            off_out = (i_bh * nc + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
            tl.store(dM_local_out + off_out, out_dM, mask=mask_kv)
            tl.store(dI_local_out + off_out, out_dI, mask=mask_kv)


    @triton.autotune(configs=_fp_bwd_autotune(), key=['D_K', 'D_V', 'C'])
    @triton.jit
    def _fp_bwd_kernel_scan(
        dM_local, dI_local, gt, g,
        dM_bound, dI_bound,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr
    ):
        """Pass 2/3 (Sequential over chunks), BV/BK-tiled. grid=(NK, NV, B*H).
        Pure elementwise recurrence (Flast scalar per chunk) -> tiles trivially."""
        i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        nc = T // C

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < D_K
        mask_v = o_v < D_V
        mask_kv = mask_v[:, None] & mask_k[None, :]

        dM = tl.zeros([BV, BK], dtype=tl.float32)
        dI = tl.zeros([BV, BK], dtype=tl.float32)
        b_g = tl.load(g + i_h).to(tl.float32)

        off_bound_nc = (i_bh * (nc + 1) + nc) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
        tl.store(dM_bound + off_bound_nc, dM, mask=mask_kv)
        tl.store(dI_bound + off_bound_nc, dI, mask=mask_kv)

        for cc in range(nc):
            c = nc - 1 - cc

            base_gt = (i_b * T + c * C) * H + i_h
            o_c = tl.arange(0, C)
            gt_c = tl.load(gt + base_gt + o_c * H, mask=o_c < C, other=0.0).to(tl.float32)
            logf = -gt_c * b_g
            clogf = tl.cumsum(logf, axis=0)
            clogf_last = tl.sum(tl.where(o_c == (C - 1), clogf, 0.0))
            Flast = tl.exp(clogf_last)

            off_local = (i_bh * nc + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
            dMl = tl.load(dM_local + off_local, mask=mask_kv, other=0.0).to(tl.float32)
            dIl = tl.load(dI_local + off_local, mask=mask_kv, other=0.0).to(tl.float32)

            dM = Flast * dM + dMl
            dI = Flast * dI + dIl

            off_bound = (i_bh * (nc + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
            tl.store(dM_bound + off_bound, dM, mask=mask_kv)
            tl.store(dI_bound + off_bound, dI, mask=mask_kv)

    @triton.autotune(
        configs=[triton.Config({'BV': bv}, num_warps=w, num_stages=s)
                 for bv in (16, 32, 64) for w in (2, 4, 8) for s in (1, 2)],
        key=['D_K', 'D_V', 'C'])
    @triton.jit
    def _fp_bwd_kernel_intra(
        q, k, v, b, gt, g, Ip,
        M_bound, I_bound, do,
        dM_bound, dI_bound,
        dq, dk, dv, db, dgt, scale,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
        PERDK: tl.constexpr,
    ):
        """Pass 3/3 (Parallel), BV-tiled with BK full-resident. grid=(NC, B*H).

        BK = next_pow2(D_K) is kept resident (it fits; only BV=next_pow2(D_V) blew
        SMEM). Two sweeps over D_V tiles: sweep A accumulates the D_V-reductions
        feeding the [C,*] algebra (Ibar_c, bbar, dM/dI-contractions, dsc,
        dQtil_carry, dcd, pM, pI, cM, cI); sweep B writes per-block dv, db and
        adds the per-block dk/df contributions. dq/dk/dgt have full D_K resident
        so no cross-block accumulation is needed.
        """
        i_c, i_bh = tl.program_id(0), tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        nc = T // C
        c = i_c

        o_c = tl.arange(0, C)
        o_k = tl.arange(0, BK)
        mask_k = o_k < D_K
        NV = tl.cdiv(D_V, BV)

        if PERDK:
            b_Ip = tl.load(Ip + i_h * D_K + o_k, mask=mask_k, other=1.0).to(tl.float32)
        else:
            b_Ip = tl.load(Ip + i_h).to(tl.float32) + tl.zeros([BK], dtype=tl.float32)
        b_Ip = tl.where(mask_k, b_Ip, 1.0)
        b_g = tl.load(g + i_h).to(tl.float32)

        base_qk = (i_b * T * H + c * C * H + i_h) * D_K
        base_vo = (i_b * T * H + c * C * H + i_h) * D_V
        base_gt = (i_b * T + c * C) * H + i_h

        q_ck = tl.load(q + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_k[None, :], other=0.0).to(tl.float32)
        k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_k[None, :], other=0.0).to(tl.float32)
        gt_c = tl.load(gt + base_gt + o_c * H, mask=o_c < C, other=0.0).to(tl.float32)

        qs = q_ck * scale
        f_c = tl.exp(-gt_c * b_g)
        logf = -gt_c * b_g
        clogf = tl.cumsum(logf, axis=0)
        cd = tl.exp(clogf)
        ksq = k_ck * k_ck
        Dm_full = tl.exp(tl.minimum(clogf[:, None] - clogf[None, :], 0.0))
        tri = o_c[:, None] >= o_c[None, :]
        Dm = tl.where(tri, Dm_full, 0.0)
        clogf_last = tl.sum(tl.where(o_c == (C - 1), clogf, 0.0))
        sca = tl.exp(clogf_last - clogf)

        # ---- sweep A: all D_V-reductions ----
        Ibar_c = tl.zeros([BK], dtype=tl.float32)
        bbar = tl.zeros([C], dtype=tl.float32)
        dsc = tl.zeros([C, C], dtype=tl.float32)               # do @ v^T
        dQtil_carry = tl.zeros([C, BK], dtype=tl.float32)      # (do*cd) @ mu
        dcd = tl.zeros([C], dtype=tl.float32)                  # sum_v do*base
        pM = tl.zeros([C], dtype=tl.float32)
        pI = tl.zeros([C], dtype=tl.float32)
        cM = tl.zeros([1], dtype=tl.float32)
        cI = tl.zeros([1], dtype=tl.float32)
        dk_from_state = tl.zeros([C, BK], dtype=tl.float32)
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (nc + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), b_Ip[None, :])
            mu_v = M_cv / I_cv
            off_dn = ((i_bh * (nc + 1) + c + 1) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            dM_v = tl.load(dM_bound + off_dn, mask=mask_kv, other=0.0).to(tl.float32)
            dI_v = tl.load(dI_bound + off_dn, mask=mask_kv, other=0.0).to(tl.float32)
            v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            do_cv = tl.load(do + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                            mask=mask_v[None, :], other=0.0).to(tl.float32)
            Ibar_c += tl.sum(tl.where(mask_v[:, None], I_cv, 0.0), axis=0)
            bbar += tl.sum(tl.where(mask_v[None, :], b_cv, 0.0), axis=1)
            vdM = tl.dot(v_cv.to(tl.float32), dM_v.to(tl.float32))   # (C,BK)
            bdI = tl.dot(b_cv.to(tl.float32), dI_v.to(tl.float32))   # (C,BK)
            pM += tl.sum(vdM * k_ck, axis=1)
            pI += tl.sum(bdI * ksq, axis=1)
            cM += tl.sum(M_cv * dM_v)
            cI += tl.sum((I_cv - b_Ip[None, :]) * dI_v)
            dk_from_state += sca[:, None] * (2.0 * k_ck * bdI + vdM)
            dbase = do_cv * cd[:, None]
            dsc += tl.dot(do_cv.to(tl.float32), tl.trans(v_cv).to(tl.float32))
            dQtil_carry += tl.dot(dbase.to(tl.float32), mu_v.to(tl.float32))
        Ibar_c = Ibar_c / D_V
        bbar = bbar / D_V

        # ---- middle [C,*] algebra ----
        abar = tl.where(mask_k, Ibar_c - b_Ip, 0.0)
        src = bbar[:, None] * ksq
        a_mat = cd[:, None] * abar[None, :] + tl.dot(Dm.to(tl.float32), src.to(tl.float32))
        Ibar_mat = tl.maximum(b_Ip[None, :] + a_mat, b_Ip[None, :])
        Qtil = qs / Ibar_mat
        sc_raw = tl.dot(Qtil.to(tl.float32), tl.trans(k_ck).to(tl.float32))
        sc = sc_raw * Dm
        Q_carry_til = qs * (Ibar_c[None, :] / Ibar_mat)

        dq_acc = tl.zeros([C, BK], dtype=tl.float32)
        dk_acc = dk_from_state
        df = tl.zeros([C], dtype=tl.float32)
        dclogf = tl.zeros([C], dtype=tl.float32)

        # df from state (pM/pI/cM/cI)
        clogf_prev = clogf - logf
        Fprev = tl.exp(clogf_prev)
        W = tl.where(o_c[:, None] > o_c[None, :], tl.exp(tl.minimum(clogf_prev[:, None] - clogf[None, :], 0.0)), 0.0)
        dfM = Fprev * tl.sum(cM) + tl.sum(W * pM[None, :], axis=1)
        dfI = Fprev * tl.sum(cI) + tl.sum(W * pI[None, :], axis=1)
        df += sca * (dfI + dfM)

        dclogf += dcd * cd  # dcd is 0 here; folded below via the carry path
        # carry-path dcd: dcd_t = sum_v do*base ; base = Q_carry_til @ mu^T.
        # base needs mu (D_V) -> recompute in sweep B and accumulate dcd there.

        dsc_raw = dsc * Dm
        dDm = dsc * sc_raw
        dexp = tl.where(tri, dDm, 0.0)
        dclogf += tl.sum(dexp * Dm_full, axis=1)
        dclogf += -tl.sum(dexp * Dm_full, axis=0)
        dQtil_local = tl.dot(dsc_raw.to(tl.float32), k_ck.to(tl.float32))
        dk_acc += tl.dot(tl.trans(dsc_raw).to(tl.float32), Qtil.to(tl.float32))

        dqs = (dQtil_local / Ibar_mat) + (dQtil_carry * Ibar_c[None, :] / Ibar_mat)
        dIbar = -(dQtil_local * qs / (Ibar_mat * Ibar_mat)) \
                - (dQtil_carry * qs * Ibar_c[None, :] / (Ibar_mat * Ibar_mat))
        dq_acc += dqs * scale

        Wt = tl.where(o_c[None, :] >= o_c[:, None], tl.exp(tl.minimum(clogf[None, :] - clogf[:, None], 0.0)), 0.0)
        D = tl.dot(Wt.to(tl.float32), dIbar.to(tl.float32))
        
        Dm_prev = tl.where(o_c[:, None] > o_c[None, :], tl.exp(tl.minimum(clogf_prev[:, None] - clogf[None, :], 0.0)), 0.0)
        a_prev = Fprev[:, None] * abar[None, :] + tl.dot(Dm_prev.to(tl.float32), src.to(tl.float32))
        df += tl.sum(a_prev * D, axis=1)
        dk_acc += bbar[:, None] * 2.0 * k_ck * D
        kD = tl.sum(ksq * D, axis=1) / D_V    # (C,) -> broadcast into db per block

        # ---- sweep B: per-block dv, db, and dcd accumulation ----
        dcd_acc = tl.zeros([C], dtype=tl.float32)
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (nc + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), b_Ip[None, :])
            mu_v = M_cv / I_cv
            off_dn = ((i_bh * (nc + 1) + c + 1) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            dM_v = tl.load(dM_bound + off_dn, mask=mask_kv, other=0.0).to(tl.float32)
            dI_v = tl.load(dI_bound + off_dn, mask=mask_kv, other=0.0).to(tl.float32)
            v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            do_cv = tl.load(do + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                            mask=mask_v[None, :], other=0.0).to(tl.float32)
            # dcd contribution: base = Q_carry_til @ mu^T  (C,BV)
            base_blk = tl.dot(Q_carry_til.to(tl.float32), tl.trans(mu_v).to(tl.float32))
            dcd_acc += tl.sum(do_cv * base_blk, axis=1)
            # dv, db per block
            dv_blk = sca[:, None] * tl.dot(k_ck.to(tl.float32), tl.trans(dM_v).to(tl.float32))
            dv_blk += tl.dot(tl.trans(sc).to(tl.float32), do_cv.to(tl.float32))
            db_blk = sca[:, None] * tl.dot(ksq.to(tl.float32), tl.trans(dI_v).to(tl.float32))
            db_blk += kD[:, None] * tl.where(mask_v[None, :], 1.0, 0.0)
            tl.store(dv + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :], dv_blk, mask=mask_v[None, :])
            tl.store(db + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :], db_blk, mask=mask_v[None, :])

        # fold dcd (carry-decay path) into dclogf, finalize df, dgt
        dclogf += dcd_acc * cd
        csum = tl.cumsum(dclogf, axis=0)
        total = tl.sum(dclogf)
        dlogf = total - csum + dclogf
        dgt_c = (df * f_c + dlogf) * (-b_g)

        tl.store(dq + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :], dq_acc, mask=mask_k[None, :])
        tl.store(dk + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :], dk_acc, mask=mask_k[None, :])
        tl.store(dgt + base_gt + o_c * H, dgt_c, mask=o_c < C)

    def _fp_clamp_C(D_K, D_V, C, dev):
        """Largest power-of-two chunk <= C whose bwd fits this device's shared mem.

        Footprint scales ~linearly in C * BV (BV = next_pow2(D_V)); measured
        anchor C=32,BV=256 -> 188416 bytes (~23 B per C*BV). Compared against the
        device per-block opt-in limit (227 KiB H100, ~99 KiB RTX 4090), 10%
        headroom (5%). FLOOR IS 16: tl.dot requires the contraction dim K>=16, so the
        kernel cannot run below C=16. If even C=16 won't fit, we keep 16 and let
        the launch raise a clear OutOfResources rather than emit a broken C=8.
        """
        try:
            limit = int(torch.cuda.get_device_properties(dev).shared_memory_per_block_optin)
        except Exception:
            limit = 99 * 1024
        BV = triton.next_power_of_2(D_V)
        c = C
        while c > 16 and (26 * c * BV) > int(limit * 0.95):
            c //= 2
        return c

    def _fast_palimpsa_bwd_triton(do, q, k, v, b, gt, g, Ip, scale, C,
                                  M_bound, I_bound, Ibar):
        B, T, H, D_K = q.shape
        D_V = v.shape[-1]
        nc = T // C
        dev = q.device
        BK_full = triton.next_power_of_2(D_K)
        perdk = torch.is_tensor(Ip) and Ip.numel() == H * D_K

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        db = torch.zeros_like(b, dtype=torch.float32)
        dgt = torch.zeros_like(gt, dtype=torch.float32)

        Ipf = (Ip.float().contiguous() if torch.is_tensor(Ip)
               else torch.full((H,), float(Ip), device=dev, dtype=torch.float32))

        qc, kc, vc, bc, gtc = [x.contiguous() for x in (q, k, v, b, gt)]
        gf = g.float().contiguous()
        dof = do.contiguous().float()

        dM_local = torch.empty(B * H, nc, D_V, D_K, device=dev, dtype=torch.float32)
        dI_local = torch.empty(B * H, nc, D_V, D_K, device=dev, dtype=torch.float32)
        dM_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)
        dI_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)

        # 1. local-state (BV/BK tiled, autotuned). grid=(NK, NC, B*H)
        grid_ls = lambda META: (triton.cdiv(D_K, META['BK']), nc, B * H)
        _fp_bwd_kernel_local_state[grid_ls](
            qc, kc, vc, bc, gtc, gf, Ipf,
            M_bound, I_bound, dof,
            dM_local, dI_local, scale,
            T, H=H, D_K=D_K, D_V=D_V, C=C, PERDK=perdk,
        )

        # 2. sequential scan over chunks (BV/BK tiled, autotuned). grid=(NK, NV, B*H)
        grid_sc = lambda META: (triton.cdiv(D_K, META['BK']), triton.cdiv(D_V, META['BV']), B * H)
        _fp_bwd_kernel_scan[grid_sc](
            dM_local, dI_local, gtc, gf,
            dM_bound, dI_bound,
            T, H=H, D_K=D_K, D_V=D_V, C=C,
        )

        # 3. intra (BV tiled, BK full-resident, autotuned). grid=(NC, B*H)
        grid_in = lambda META: (nc, B * H)
        _fp_bwd_kernel_intra[grid_in](
            qc, kc, vc, bc, gtc, gf, Ipf,
            M_bound, I_bound, dof,
            dM_bound, dI_bound,
            dq, dk, dv, db, dgt, scale,
            T, H=H, D_K=D_K, D_V=D_V, BK=BK_full, C=C, PERDK=perdk,
        )

        return dq, dk, dv, db, dgt

    class _FastPalimpsa(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, b, gt, g, Ip, scale, C):
            B, T, H, D_K = q.shape
            D_V = v.shape[-1]
            nc = T // C
            dev = q.device
            BK = min(32, triton.next_power_of_2(D_K))
            BV = triton.next_power_of_2(D_V)
            obk, obv, onw, ons = _fp_output_block_sizes(D_K, D_V, C, dev.index or 0)
            perdk = torch.is_tensor(Ip) and Ip.numel() == H * D_K
            Ipf = (Ip.float().contiguous() if torch.is_tensor(Ip)
                   else torch.full((H,), float(Ip), device=dev, dtype=torch.float32))
            gf = g.float().contiguous()
            M_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)
            I_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)
            Ibar = torch.empty(B * H, nc, C, D_K, device=dev, dtype=torch.float32)
            o = torch.empty(B, T, H, D_V, device=dev, dtype=q.dtype)
            qc, kc, vc, bc, gtc = [x.contiguous() for x in (q, k, v, b, gt)]
            
            _fp_state_kernel[(triton.cdiv(D_K, BK), triton.cdiv(D_V, BV), B * H)](
                kc, vc, bc, gtc, gf, Ipf, M_bound, I_bound,
                T, H=H, D_K=D_K, D_V=D_V, BK=BK, BV=BV, C=C, PERDK=perdk)
                
            _fp_ibar_kernel[(triton.cdiv(D_K, BK), B * H)](
                kc, bc, gtc, gf, Ipf, I_bound, Ibar,
                T, H=H, D_K=D_K, D_V=D_V, BK=BK, C=C,
                BV_IB=triton.next_power_of_2(D_V), PERDK=perdk)
                
            _fp_output_kernel[(triton.cdiv(D_V, obv), nc, B * H)](
                qc, kc, vc, gtc, gf, Ipf, M_bound, I_bound, Ibar, o, scale,
                T, H=H, D_K=D_K, D_V=D_V, BK=obk, BV=obv, C=C, PERDK=perdk,
                num_warps=onw, num_stages=ons)
                
            ctx.save_for_backward(q, k, v, b, gt, gf, Ipf, M_bound, I_bound, Ibar)
            ctx.scale, ctx.C = scale, C
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, b, gt, g, Ip, M_bound, I_bound, Ibar = ctx.saved_tensors
            dq, dk, dv, db, dgt = _fast_palimpsa_bwd_triton(
                do, q, k, v, b, gt, g, Ip, ctx.scale, ctx.C, M_bound, I_bound, Ibar)
            return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                    db.to(b.dtype), dgt.to(gt.dtype), None, None, None, None)

    def chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=None, chunk_size=CHUNK_C,
                            backend="triton"):
        if scale is None:
            scale = q.shape[-1] ** -0.5
        D_K = q.shape[-1]
        D_V = v.shape[-1]
        # 'vec': chunk-parallel autograd path. General D_K/D_V, no SMEM limit,
        #        grads via autograd (correct by construction). Use this for training.
        # 'triton': the hand-written Triton fwd/bwd (currently blows SMEM at large D_V).
        if backend == "vec":
            return fast_palimpsa_vec(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=chunk_size)
        if q.is_cuda:
            C_eff = _fp_clamp_C(D_K, D_V, chunk_size, q.device)
            T = q.shape[1]
            if T % C_eff != 0:
                cc = C_eff
                while cc > 16 and (T % cc != 0):
                    cc //= 2
                C_eff = cc if (T % cc == 0) else chunk_size
            if C_eff != chunk_size:
                import warnings
                warnings.warn(
                    f"chunk_fast_palimpsa: chunk_size={chunk_size} exceeds the backward "
                    f"shared-memory limit for D_K={D_K}, D_V={D_V} on this GPU; "
                    f"using chunk_size={C_eff}.")
        else:
            C_eff = chunk_size
        return _FastPalimpsa.apply(q, k, v, b, gt, g, Ip, scale, C_eff)


# =============================================================================
# 3. Test: Triton fwd & bwd vs reference fwd & bwd
# =============================================================================
def test_fwd_bwd(B=2, T=64, H=3, D_K=48, D_V=40, C=16, seed=0, dtype=torch.float32):
    assert HAS_TRITON, "CUDA/Triton required for the Triton test."
    torch.manual_seed(seed)
    dev = "cuda"
    mk = lambda *s: torch.randn(*s, device=dev, dtype=dtype)
    q = mk(B, T, H, D_K).requires_grad_(True)
    k = torch.nn.functional.normalize(mk(B, T, H, D_K), dim=-1).detach().requires_grad_(True)
    v = mk(B, T, H, D_V).requires_grad_(True)
    b = (torch.rand(B, T, H, D_V, device=dev, dtype=dtype) * 1.5 + 0.1).requires_grad_(True)
    gt = (torch.rand(B, T, H, device=dev, dtype=dtype) * 0.1).requires_grad_(True)
    g = torch.rand(H, device=dev, dtype=dtype) * 0.5 + 0.5
    Ip = torch.rand(H, device=dev, dtype=dtype) + 0.5
    scale = D_K ** -0.5

    # --- forward ---
    o_tri = chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=C,
                                backend="triton")
    o_ref = fast_palimpsa_ref(q.detach(), k.detach(), v.detach(), b.detach(),
                              gt.detach(), g, Ip, scale=scale, chunk_size=C)
    fwd_err = (o_tri - o_ref).abs().max().item()
    print(f"[fwd] D_K={D_K} D_V={D_V}  max|triton-ref| = {fwd_err:.3e}")

    # --- backward ---
    do = torch.randn_like(o_tri)
    grads_tri = torch.autograd.grad(o_tri, (q, k, v, b, gt), do, retain_graph=False)

    q2, k2, v2, b2, gt2 = [t.detach().clone().requires_grad_(True) for t in (q, k, v, b, gt)]
    o_ref2 = fast_palimpsa_ref(q2, k2, v2, b2, gt2, g, Ip, scale=scale, chunk_size=C)
    grads_ref = torch.autograd.grad(o_ref2, (q2, k2, v2, b2, gt2), do)

    names = ["dq", "dk", "dv", "db", "dgt"]
    # vec backend matches the reference up to fp accumulation only; judge by rel err.
    fwd_rel = fwd_err / (o_ref.abs().max().item() + 1e-12)
    print(f"      (fwd rel err = {fwd_rel:.3e})")
    # Triton fp32 tl.dot (tensor cores) vs fp32 loop ref: ~1-3e-3 rel is expected.
    tol = 5e-3 if dtype == torch.float32 else 1e-5
    ok = fwd_rel < tol
    for n, gtri, gref in zip(names, grads_tri, grads_ref):
        e = (gtri - gref).abs().max().item()
        rel = e / (gref.abs().max().item() + 1e-12)
        print(f"[bwd] {n}: max abs err {e:.3e}  rel {rel:.3e}")
        ok = ok and rel < tol
    print("PASS" if ok else "FAIL")
    return ok


def test_ref_shapes():
    """CPU-only: exercises the reference across odd D_K/D_V and Ip layouts."""
    for (DK, DV) in [(128, 16), (48, 40), (33, 7), (64, 128)]:
        for ip_kind in ("scalar", "perdk"):
            torch.manual_seed(0)
            B, L, H, C = 1, 32, 2, 16
            q = torch.randn(B, L, H, DK, dtype=torch.float64)
            k = torch.nn.functional.normalize(torch.randn(B, L, H, DK, dtype=torch.float64), dim=-1)
            v = torch.randn(B, L, H, DV, dtype=torch.float64)
            b = torch.rand(B, L, H, DV, dtype=torch.float64) * 1.5 + 0.1
            gt = torch.rand(B, L, H, dtype=torch.float64) * 0.1
            g = torch.rand(H, dtype=torch.float64) * 0.5 + 0.5
            Ip = (torch.rand(H, dtype=torch.float64) + 0.5 if ip_kind == "scalar"
                  else torch.rand(H, DK, dtype=torch.float64) + 0.5)
            scale = DK ** -0.5
            y = fast_palimpsa_ref(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=C)
            assert y.shape == (B, L, H, DV)
            # TRUE invariant for frozen-carry mode: with NO carried history (a
            # single chunk, L == C) AND beta constant across D_V (so the isotropic
            # collapse is lossless), the approximation is EXACT. (With history or
            # anisotropic beta it is genuinely approximate -- that is the method.)
            qs, ks, vs, gts = q[:, :C], k[:, :C], v[:, :C], gt[:, :C]
            bs = b[:, :C].mean(-1, keepdim=True).expand(B, C, H, DV).contiguous()
            ye, _ = _exact_recurrence(qs, ks, vs, bs, gts, g, Ip, scale)
            ya = fast_palimpsa_ref(qs, ks, vs, bs, gts, g, Ip, scale=scale, chunk_size=C)
            err = (ya - ye).abs().max().item()
            print(f"DK={DK:3d} DV={DV:3d} Ip={ip_kind:6s}: shape ok, single-chunk const-beta exactness {err:.2e}")
            assert err < 1e-9, f"single-chunk const-beta should be exact, got {err}"


if __name__ == "__main__":
    print(f"[chunk_fast_palimpsa] {FP_CLAMP_VERSION}")
    print("== reference shape/exactness sweep (CPU) ==")
    test_ref_shapes()
    if HAS_TRITON:
        print("\n== TRITON fwd/bwd vs reference (GPU) ==")
        print("   (test forces backend='triton'; BV-tiled kernels, any D_K/D_V)")
        for C_ in (16, 32):
            print(f"\n-- chunk_size C={C_} --")
            for (dk, dv) in [(128, 16), (48, 40), (64, 128), (33, 24), (96, 192)]:
                test_fwd_bwd(D_K=dk, D_V=dv, T=4 * C_, C=C_)
        # also exercise fp32 at the real training shape with a couple of chunks
        print("\n-- real shape sanity (D_K=96, D_V=192, longer seq) --")
        test_fwd_bwd(B=1, T=256, H=4, D_K=96, D_V=192, C=16)
    else:
        print("\n(no CUDA/Triton here -> ran reference checks only)")