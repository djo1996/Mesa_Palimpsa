# -*- coding: utf-8 -*-
# chunk_fast_palimpsa_v2.py
#
# Fast Palimpsa V2: chunked RANK-1 FACTORIZED-precision approximation of Palimpsa.
#   The per-token precision tensor I_{t,v,k} (full D_V x D_K) is approximated by a
#   separable rank-1 form built from three isotropic marginals:
#
#         I_{t,v,k}  ~=  Ibar_V[t,v] * Ibar_K[t,k] / Ibar_S[t]
#
#   where (all evolved with the full alpha-decay + (1-f) Ip prior + beta k^2 dynamics)
#     * Ibar_K[t] in R^{D_K}  collapses I over D_V   (driven by betabar_K = mean_v b)
#     * Ibar_V[t] in R^{D_V}  collapses I over D_K   (driven by ksqbar_V = mean_k k^2)
#     * Ibar_S[t]            = mean_k Ibar_K[t]      (scalar normaliser)
#   This is strictly richer than V1 (which used Ibar_K alone): V2 also captures the
#   D_V-anisotropy of the precision. With beta constant over D_V the V-collapse is
#   lossless and V2 == V1; with beta constant over BOTH axes the rank-1 form is exact.
#
#   * State carried across chunks is EXACT, full D_V x D_K (identical to V1).
#   * LOCAL read:  reader-side Qtil = q/Ibar_K, then a final (Ibar_S/Ibar_V) multiplier.
#   * CARRY read:  frozen mu_c = M_c/I_c with Q reweight (Ibar_c_K/Ibar_K) and a
#                  carry_ratio = (Ibar_c_V/Ibar_V)*(Ibar_S/Ibar_c_S) multiplier.
#
# This file contains:
#   1. fast_palimpsa_v2_ref    -- differentiable PyTorch reference (the contract).
#   2. fast_palimpsa_v2_vec    -- vectorized closed-form (autograd) training path.
#   3. chunk_fast_palimpsa     -- Triton autograd Function (tiled, any D_K/D_V).
#   4. test_fwd_bwd()          -- compares Triton fwd AND bwd against the reference.
#
# SCALAR Ip ONLY (Ip is (H,)): the rank-1 factorization assumes a scalar prior.
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
FP_CLAMP_VERSION = "V2-rank1-factorized-v1"   # bump to confirm the right file is loaded


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
        # V2 persistent adds ibarV and carry_ratio (each C*BV) on top of V1's
        # score,Dmask (C*C) and carry,v (C*BV).
        persistent = (2 * C * C + 4 * C * BV) * 4
        return persistent + stages * per_iter

    for BK in (min(64, triton.next_power_of_2(D_K)), 32, 16):
        for stages in (3, 2, 1):
            if est(BK, stages) <= limit:
                num_warps = 4 if BV >= 64 else 2
                return BK, BV, num_warps, stages
    # Last resort: smallest BK, single stage.
    return 16, BV, 2, 1


# =============================================================================
# 1. Reference (differentiable). SCALAR Ip only.
# =============================================================================
def _resolve_Ip_scalar(Ip, H, dev, dt):
    """V2 is scalar-Ip only. Returns Ip as (H,) float tensor."""
    if not torch.is_tensor(Ip):
        Ip = torch.full((H,), float(Ip), device=dev, dtype=dt)
    Ip = Ip.to(device=dev, dtype=dt)
    if Ip.numel() != H:
        raise ValueError("fast_palimpsa_v2 supports scalar-per-head Ip (H,) only.")
    return Ip


def _exact_recurrence(q, k, v, b, gt, g, Ip, scale):
    """Token-exact Palimpsa ground truth (scalar Ip). Differentiable."""
    B, L, H, DK = q.shape
    DV = v.shape[-1]
    dev, dt = q.device, q.dtype
    Ip = _resolve_Ip_scalar(Ip, H, dev, dt)
    Ipv = Ip.view(1, H, 1, 1)
    q = q * scale
    M = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
    I = Ipv.expand(B, H, DV, DK).clone()
    ys = []
    for t in range(L):
        f = torch.exp(-(gt[:, t] * g.view(1, H))).view(B, H, 1, 1)
        kt = k[:, t].view(B, H, 1, DK)
        vt = v[:, t].view(B, H, DV, 1)
        bt = b[:, t].view(B, H, DV, 1)
        I = f * I + (1 - f) * Ipv + bt * (kt * kt)
        M = f * M + vt * kt
        mu = M / I
        qt = q[:, t].view(B, H, 1, DK)
        ys.append((mu * qt).sum(-1))
    return torch.stack(ys, 1)


def fast_palimpsa_ref(q, k, v, b, gt, g, Ip, scale=None, chunk_size=CHUNK_C,
                      output_uncertainty=False):
    """V2 rank-1 factorized approximation (frozen carry). Differentiable. The contract.

    I_{t,v,k} ~= Ibar_V[t,v] * Ibar_K[t,k] / Ibar_S[t].
    """
    B, L, H, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    assert L % C == 0
    nc = L // C
    if scale is None:
        scale = DK ** -0.5
    dev, dt = q.device, q.dtype
    Ip = _resolve_Ip_scalar(Ip, H, dev, dt)
    Ip_K = Ip.view(1, H, 1)
    Ip_V = Ip.view(1, H, 1)
    Ipv_full = Ip.view(1, H, 1, 1).expand(B, H, DV, DK)

    qs = q * scale
    M_c = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
    I_c = Ipv_full.clone()

    y_out = torch.zeros(B, L, H, DV, dtype=dt, device=dev)
    if output_uncertainty:
        yvar_out = torch.zeros(B, L, H, DV, dtype=dt, device=dev)

    for c in range(nc):
        sl = slice(c * C, (c + 1) * C)
        kc, vc, bc, qc, gc = k[:, sl], v[:, sl], b[:, sl], qs[:, sl], gt[:, sl]
        f = torch.exp(-(gc * g.view(1, 1, H)))
        logf = torch.log(f.clamp_min(1e-30))
        clogf = torch.cumsum(logf, dim=1)

        Ibar_c_K = I_c.mean(2)                 # (B,H,DK)
        Ibar_c_V = I_c.mean(3)                 # (B,H,DV)
        Ibar_c_S = I_c.mean(dim=(2, 3))        # (B,H)

        abar_K_prev = Ibar_c_K - Ip_K
        abar_V_prev = Ibar_c_V - Ip_V
        betabar_K = bc.mean(-1)                # (B,C,H)
        ksq = kc * kc
        ksqbar_V = ksq.mean(-1)                # (B,C,H)

        Ibar_K = torch.empty(B, C, H, DK, dtype=dt, device=dev)
        Ibar_V = torch.empty(B, C, H, DV, dtype=dt, device=dev)
        a_K_prev, a_V_prev = abar_K_prev, abar_V_prev
        for t in range(C):
            ft = f[:, t].unsqueeze(-1)
            a_K_prev = ft * a_K_prev + betabar_K[:, t].unsqueeze(-1) * ksq[:, t]
            Ibar_K[:, t] = Ip_K + a_K_prev
            a_V_prev = ft * a_V_prev + bc[:, t] * ksqbar_V[:, t].unsqueeze(-1)
            Ibar_V[:, t] = Ip_V + a_V_prev

        Ibar_S = Ibar_K.mean(-1)               # (B,C,H)

        # local (rank-1 scaling split: reader-side Qtil=q/Ibar_K, final Ibar_S/Ibar_V)
        Qtil = qc / Ibar_K
        score = torch.einsum('bthd,bshd->btsh', Qtil, kc)
        Dmask = torch.exp(clogf.unsqueeze(2) - clogf.unsqueeze(1))
        tri = torch.tril(torch.ones(C, C, device=dev, dtype=dt)).view(1, C, C, 1)
        score = score * Dmask * tri
        y_local = torch.einsum('btsh,bshv->bthv', score, vc)
        y_local = y_local * (Ibar_S.unsqueeze(-1) / Ibar_V)

        # carry (frozen mu_c, Q reweight Ibar_c_K/Ibar_K, ratio (Ibar_c_V/Ibar_V)(Ibar_S/Ibar_c_S))
        mu_c = M_c / I_c
        Q_carry_til = qc * (Ibar_c_K.unsqueeze(1) / Ibar_K)
        base = torch.einsum('bhvd,bthd->bthv', mu_c, Q_carry_til)
        carry_ratio = (Ibar_c_V.unsqueeze(1) / Ibar_V) * (Ibar_S / Ibar_c_S.unsqueeze(1)).unsqueeze(-1)
        base = base * carry_ratio
        carry_decay = torch.exp(clogf).unsqueeze(-1)
        y_out[:, sl] = y_local + base * carry_decay

        if output_uncertainty:
            qc_sq = qc * qc
            var_base = torch.einsum('bthd,bthd->bth', qc_sq, 1.0 / Ibar_K)
            yvar_out[:, sl] = var_base.unsqueeze(-1) * (Ibar_S.unsqueeze(-1) / Ibar_V)

        # exact boundary update (identical to V1)
        I_new, M_new = I_c.clone(), M_c.clone()
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
    """V2, fully vectorized (closed-form intra-chunk scans). Autograd-diff.

    Same math as fast_palimpsa_ref but the two `for t in range(C)` token loops
    (the dual Ibar_K/Ibar_V scan AND the exact boundary update) are replaced by
    cumulative-decay matmuls. Differentiable by autograd, any D_K/D_V, no SMEM
    limit. This is the training path; it matches the loop reference fwd + all
    grads to fp64 machine precision.
    """
    B, L, H, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    assert L % C == 0, f"L={L} not divisible by C={C}"
    nc = L // C
    if scale is None:
        scale = DK ** -0.5
    dev, dt = q.device, q.dtype
    Ip = _resolve_Ip_scalar(Ip, H, dev, dt)
    Ip_K = Ip.view(1, H, 1)
    Ip_V = Ip.view(1, H, 1)

    qs = q * scale
    M_c = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
    I_c = Ip.view(1, H, 1, 1).expand(B, H, DV, DK).clone()

    triCC = torch.tril(torch.ones(C, C, device=dev, dtype=dt)).view(1, C, C, 1)
    y_out = torch.zeros(B, L, H, DV, dtype=dt, device=dev)

    for c in range(nc):
        sl = slice(c * C, (c + 1) * C)
        kc, vc, bc, qc, gc = k[:, sl], v[:, sl], b[:, sl], qs[:, sl], gt[:, sl]
        f = torch.exp(-(gc * g.view(1, 1, H)))
        logf = torch.log(f.clamp_min(1e-30))
        clogf = torch.cumsum(logf, dim=1)
        cd = torch.exp(clogf)

        Ibar_c_K = I_c.mean(2)
        Ibar_c_V = I_c.mean(3)
        Ibar_c_S = I_c.mean(dim=(2, 3))
        abar_K = Ibar_c_K - Ip_K
        abar_V = Ibar_c_V - Ip_V
        betabar_K = bc.mean(-1)
        ksq = kc * kc
        ksqbar_V = ksq.mean(-1)

        Dm = torch.exp(clogf.unsqueeze(2) - clogf.unsqueeze(1)) * triCC   # (B,C,C,H)[t,i]

        srcK = betabar_K.unsqueeze(-1) * ksq                    # (B,C,H,DK)[i]
        aK = cd.unsqueeze(-1) * abar_K.unsqueeze(1) + torch.einsum('btih,bihd->bthd', Dm, srcK)
        Ibar_K = Ip_K.unsqueeze(1) + aK

        srcV = bc * ksqbar_V.unsqueeze(-1)                      # (B,C,H,DV)[i]
        aV = cd.unsqueeze(-1) * abar_V.unsqueeze(1) + torch.einsum('btih,bihv->bthv', Dm, srcV)
        Ibar_V = Ip_V.unsqueeze(1) + aV

        Ibar_S = Ibar_K.mean(-1)

        Qtil = qc / Ibar_K
        score = torch.einsum('bthd,bshd->btsh', Qtil, kc)
        Dmask = torch.exp(clogf.unsqueeze(2) - clogf.unsqueeze(1))
        score = score * Dmask * triCC
        y_local = torch.einsum('btsh,bshv->bthv', score, vc)
        y_local = y_local * (Ibar_S.unsqueeze(-1) / Ibar_V)

        mu_c = M_c / I_c
        Q_carry_til = qc * (Ibar_c_K.unsqueeze(1) / Ibar_K)
        base = torch.einsum('bhvd,bthd->bthv', mu_c, Q_carry_til)
        carry_ratio = (Ibar_c_V.unsqueeze(1) / Ibar_V) * (Ibar_S / Ibar_c_S.unsqueeze(1)).unsqueeze(-1)
        base = base * carry_ratio
        y_out[:, sl] = y_local + base * cd.unsqueeze(-1)

        prodF = torch.exp(clogf[:, -1])
        w = torch.exp(clogf[:, -1:].expand(B, C, H) - clogf)
        pf = prodF.view(B, H, 1, 1)
        M_c = pf * M_c + torch.einsum('bthv,bthd->bhvd', w.unsqueeze(-1) * vc, kc)
        I_c = (pf * I_c + (1 - pf) * Ip.view(1, H, 1, 1)
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
        Ibar_out,                    # (B,H,nc,C,DK) Ibar_K per-token precision
        IbarV_out,                   # (B,H,nc,C,DV) Ibar_V per-token precision
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, C: tl.constexpr,
        BV_IB: tl.constexpr, BK_IB: tl.constexpr,
    ):
        # One program per (b,h,k-block).
        #   * Every k-block evolves Ibar_K[t] over its BK slice (collapsing I_bound
        #     over D_V), writing Ibar_out.
        #   * The i_k==0 program ALSO evolves Ibar_V[t] over the full D_V axis
        #     (collapsing I_bound over D_K and using ksqbar_V = mean_k k^2 with a
        #     full-D_K resident load), writing IbarV_out. Ibar_V is k-block
        #     independent so a single program owns it.
        # Scalar Ip only.
        i_k, i_bh = tl.program_id(0), tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        nc = T // C
        o_c = tl.arange(0, C)
        o_k = i_k * BK + tl.arange(0, BK)
        mask_k = o_k < D_K

        b_Ip = tl.load(Ip + i_h).to(tl.float32)
        b_Ip_k = b_Ip + tl.zeros([BK], dtype=tl.float32)
        b_g = tl.load(g + i_h).to(tl.float32)

        o_v = tl.arange(0, BV_IB)
        mask_v = o_v < D_V
        o_kf = tl.arange(0, BK_IB)            # full-D_K range for ksqbar_V / V-collapse
        mask_kf = o_kf < D_K

        for c in range(nc):
            base_qk = (i_b * T * H + c * C * H + i_h) * D_K
            base_vo = (i_b * T * H + c * C * H + i_h) * D_V
            base_gt = (i_b * T + c * C) * H + i_h
            gt_c = tl.load(gt + base_gt + o_c * H, mask=o_c < C, other=0.0).to(tl.float32)
            logf = -gt_c * b_g
            clogf = tl.cumsum(logf, axis=0)
            cd = tl.exp(clogf)
            Dm = tl.where(o_c[:, None] >= o_c[None, :],
                          tl.exp(clogf[:, None] - clogf[None, :]), 0.0)

            # ---- Ibar_K[t] over this BK slice ----
            off_iv = ((i_bh * (nc + 1) + c) * D_V * D_K
                      + o_v[:, None] * D_K + o_k[None, :])
            I_c = tl.load(I_bound + off_iv, mask=(mask_v[:, None] & mask_k[None, :]),
                          other=0.0).to(tl.float32)
            Ibar_c_K = tl.sum(tl.where(mask_v[:, None], I_c, 0.0), axis=0) / D_V
            abar_K = Ibar_c_K - b_Ip_k
            k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_k[None, :], other=0.0).to(tl.float32)
            ksq = k_ck * k_ck
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            bbar = tl.sum(b_cv, axis=1) / D_V                 # betabar_K (C,)
            aK = cd[:, None] * abar_K[None, :] + tl.dot(
                Dm.to(tl.float32), (bbar[:, None] * ksq).to(tl.float32))
            Ibar_K = b_Ip_k[None, :] + aK                     # (C,BK)
            off_outK = ((i_bh * nc + c) * C + o_c[:, None]) * D_K + o_k[None, :]
            tl.store(Ibar_out + off_outK, Ibar_K, mask=mask_k[None, :])

            # ---- Ibar_V[t] over the full D_V axis (i_k==0 only) ----
            if i_k == 0:
                # collapse I_bound[c] over D_K -> Ibar_c_V (BV_IB,)
                off_ivk = ((i_bh * (nc + 1) + c) * D_V * D_K
                           + o_v[:, None] * D_K + o_kf[None, :])
                I_cvk = tl.load(I_bound + off_ivk,
                                mask=(mask_v[:, None] & mask_kf[None, :]),
                                other=0.0).to(tl.float32)
                Ibar_c_V = tl.sum(tl.where(mask_kf[None, :], I_cvk, 0.0), axis=1) / D_K
                abar_V = Ibar_c_V - b_Ip                      # (BV_IB,)
                # ksqbar_V[t] = mean_k k^2  (full-D_K resident load)
                k_full = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_kf[None, :],
                                 mask=mask_kf[None, :], other=0.0).to(tl.float32)
                ksqbar_V = tl.sum(tl.where(mask_kf[None, :], k_full * k_full, 0.0), axis=1) / D_K  # (C,)
                b_full = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                                 mask=mask_v[None, :], other=0.0).to(tl.float32)
                srcV = ksqbar_V[:, None] * b_full             # (C,BV_IB) [i,v]
                aV = cd[:, None] * abar_V[None, :] + tl.dot(Dm.to(tl.float32), srcV.to(tl.float32))
                Ibar_V = b_Ip + aV                            # (C,BV_IB)
                off_outV = ((i_bh * nc + c) * C + o_c[:, None]) * D_V + o_v[None, :]
                tl.store(IbarV_out + off_outV, Ibar_V, mask=mask_v[None, :])

    @triton.jit
    def _fp_output_kernel(
        q, k, v, gt, g,
        M_bound, I_bound, Ibar, IbarV,
        o, scale,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
    ):
        # One program per (b,h,chunk,v-block). Accumulates local score over BK
        # blocks, then local @ v and carry @ q, then applies the V2 rank-1
        # multipliers (Ibar_S/Ibar_V on local; (Ibar_c_V/Ibar_V)(Ibar_S/Ibar_c_S)
        # on carry). Ibar_S = mean_k Ibar_K accumulated across the k-loop;
        # Ibar_c_S = mean_vk I_c likewise.
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
        Dmask = tl.exp(clogf[:, None] - clogf[None, :])
        causal = o_c[:, None] >= o_c[None, :]
        Dmask = tl.where(causal, Dmask, 0.0)              # (C,C)

        base_qk = (i_b * T * H + i_c * C * H + i_h) * D_K
        base_vo = (i_b * T * H + i_c * C * H + i_h) * D_V

        v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                       mask=mask_v[None, :], other=0.0).to(tl.float32)
        # Ibar_V for this (chunk, v-block): (C,BV)
        ibarV = tl.load(IbarV + ((i_bh * nc + i_c) * C + o_c[:, None]) * D_V + o_v[None, :],
                        mask=mask_v[None, :], other=1.0).to(tl.float32)
        ibarV = tl.maximum(ibarV, 1e-5)

        score = tl.zeros([C, C], dtype=tl.float32)
        carry = tl.zeros([C, BV], dtype=tl.float32)
        sumIbarK = tl.zeros([C], dtype=tl.float32)        # -> Ibar_S = sumIbarK/D_K
        sum_Ic = tl.zeros([1], dtype=tl.float32)          # -> Ibar_c_S = sum_Ic/(D_V*D_K)
        # Ibar_c_V (D_V-axis, k-collapsed): accumulate sum_k I_c over this v-block
        Ibar_c_V = tl.zeros([BV], dtype=tl.float32)
        NK = tl.cdiv(D_K, BK)
        for i_k in range(NK):
            o_k = i_k * BK + tl.arange(0, BK)
            mask_k = o_k < D_K
            mask_kv = mask_v[:, None] & mask_k[None, :]
            q_ck = tl.load(q + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_k[None, :], other=0.0).to(tl.float32) * scale
            k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                           mask=mask_k[None, :], other=0.0).to(tl.float32)
            ibar = tl.load(Ibar + ((i_bh * nc + i_c) * C + o_c[:, None]) * D_K + o_k[None, :],
                           mask=mask_k[None, :], other=0.0).to(tl.float32)
            ibar_cl = tl.maximum(ibar, 1e-5)
            Qtil = q_ck / ibar_cl
            score += tl.dot(Qtil.to(tl.float32), tl.trans(k_ck))
            sumIbarK += tl.sum(tl.where(mask_k[None, :], ibar, 0.0), axis=1)  # Ibar_K masked-out=0

            # carry term: Qtil_carry @ mu_c^T over this k-block
            off_st = ((i_bh * (nc + 1) + i_c) * D_V * D_K
                      + o_v[:, None] * D_K + o_k[None, :])
            M_c = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_c = tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32)
            I_c = tl.maximum(I_c, 1e-5)

            # Ibar_c_K (k-axis, v-collapsed) for the carry Q reweight
            Ibar_c_K = tl.sum(tl.where(mask_v[:, None], I_c, 0.0), axis=0) / D_V
            Q_carry_til = q_ck * (Ibar_c_K[None, :] / ibar_cl)
            mu_c = M_c / I_c
            carry += tl.dot(Q_carry_til.to(tl.float32), tl.trans(mu_c))

            # accumulate Ibar_c_V (sum over k of I_c) and global sum_Ic
            Ibar_c_V += tl.sum(tl.where(mask_kv, I_c, 0.0), axis=1)
            sum_Ic += tl.sum(tl.where(mask_kv, I_c, 0.0))

        Ibar_S = sumIbarK / D_K                           # (C,)
        Ibar_S = tl.maximum(Ibar_S, 1e-5)
        Ibar_c_V = Ibar_c_V / D_K                         # (BV,)
        Ibar_c_S = tl.maximum(tl.sum(sum_Ic) / (D_V * D_K), 1e-5)

        score = score * Dmask
        y_local = tl.dot(score.to(tl.float32), v_cv)
        # V2 local multiplier: Ibar_S/Ibar_V
        y_local = y_local * (Ibar_S[:, None] / ibarV)
        # V2 carry multiplier: (Ibar_c_V/Ibar_V)*(Ibar_S/Ibar_c_S)
        carry_ratio = (Ibar_c_V[None, :] / ibarV) * (Ibar_S[:, None] / Ibar_c_S)
        y = y_local + carry * carry_ratio * carry_decay[:, None]
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
        # V2 is scalar-Ip only; the output kernel reconstructs Ibar_c_S over its
        # v-block so it REQUIRES full-D_V residency (obv >= D_V, i.e. NV==1).
        assert obv >= D_V, "V2 output kernel needs BV>=D_V (full value residency)."
        if not torch.is_tensor(Ip):
            Ip = torch.full((H,), float(Ip), device=dev, dtype=torch.float32)
        assert Ip.numel() == H, "fast_palimpsa_v2 supports scalar-per-head Ip (H,) only."
        Ip = Ip.float().contiguous()
        g = g.float().contiguous()

        M_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)
        I_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)
        Ibar = torch.empty(B * H, nc, C, D_K, device=dev, dtype=torch.float32)
        IbarV = torch.empty(B * H, nc, C, D_V, device=dev, dtype=torch.float32)
        o = torch.empty(B, T, H, D_V, device=dev, dtype=q.dtype)

        qc, kc, vc, bc, gtc = [x.contiguous() for x in (q, k, v, b, gt)]

        _fp_state_kernel[(triton.cdiv(D_K, BK), triton.cdiv(D_V, BV), B * H)](
            kc, vc, bc, gtc, g, Ip, M_bound, I_bound,
            T, H=H, D_K=D_K, D_V=D_V, BK=BK, BV=BV, C=C, PERDK=False)
        _fp_ibar_kernel[(triton.cdiv(D_K, BK), B * H)](
            kc, bc, gtc, g, Ip, I_bound, Ibar, IbarV,
            T, H=H, D_K=D_K, D_V=D_V, BK=BK, C=C,
            BV_IB=triton.next_power_of_2(D_V),
            BK_IB=triton.next_power_of_2(D_K))
        _fp_output_kernel[(triton.cdiv(D_V, obv), nc, B * H)](
            qc, kc, vc, gtc, g, M_bound, I_bound, Ibar, IbarV, o, scale,
            T, H=H, D_K=D_K, D_V=D_V, BK=obk, BV=obv, C=C,
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

    def _fp_bwd_v_autotune():
        # BK is full-resident (passed as constexpr by the driver), so autotune BV only.
        return [
            triton.Config({'BV': bv}, num_warps=w, num_stages=s)
            for bv in (16, 32, 64)
            for w in (2, 4, 8)
            for s in (1, 2)
        ]

    # =====================================================================
    # V2 backward: clean 5-stage partition (no duplicated math across kernels).
    #   1   local_state : FULL read VJP  -> dM_local, dI_local(complete EXCEPT the
    #                     Ibar_V-scan entry term dabar_V), dq_r,dk_r,dv_r,db_r,dgt_r,
    #                     and dIbarV (upstream grad of Ibar_V).
    #   1.5 ibarv       : consume dIbarV -> ADD dabar_V into dI_local (before scan);
    #                     write dk_V, db_V, dgt_V.
    #   2   scan        : Flast passthrough (V1-IDENTICAL).
    #   3   intra       : boundary-update VJP ONLY -> dk_b,dv_b,db_b,dgt_b (V1 math).
    #   driver sums:  dq=dq_r ; dk=dk_r+dk_V+dk_b ; dv=dv_r+dv_b ;
    #                 db=db_r+db_V+db_b ; dgt=dgt_r+dgt_V+dgt_b.
    # All bwd kernels: BK=next_pow2(D_K) FULL-RESIDENT, tile only D_V (BV). Scalar Ip.
    # =====================================================================

    @triton.autotune(configs=_fp_bwd_v_autotune(), key=['D_K', 'D_V', 'C'])
    @triton.jit
    def _fp_bwd_kernel_local_state(
        q, k, v, b, gt, g, Ip,
        M_bound, I_bound, Ibar, IbarV, do,
        dM_local_out, dI_carry_out, dI_K_out, dIbarV_out,
        dq_out, dk_out, dv_out, db_out, dgt_out, scale,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
    ):
        i_c, i_bh = tl.program_id(0), tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        nc = T // C
        c = i_c
        o_c = tl.arange(0, C)
        o_k = tl.arange(0, BK)
        mask_k = o_k < D_K
        NV = tl.cdiv(D_V, BV)

        b_Ip = tl.load(Ip + i_h).to(tl.float32)
        b_g = tl.load(g + i_h).to(tl.float32)
        base_qk = (i_b * T * H + c * C * H + i_h) * D_K
        base_vo = (i_b * T * H + c * C * H + i_h) * D_V
        base_gt = (i_b * T + c * C) * H + i_h

        gt_c = tl.load(gt + base_gt + o_c * H, mask=o_c < C, other=0.0).to(tl.float32)
        logf = -gt_c * b_g
        clogf = tl.cumsum(logf, axis=0)
        cd = tl.exp(clogf)
        tri = o_c[:, None] >= o_c[None, :]
        Dm = tl.where(tri, tl.exp(clogf[:, None] - clogf[None, :]), 0.0)

        q_ck = tl.load(q + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_k[None, :], other=0.0).to(tl.float32)
        k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_k[None, :], other=0.0).to(tl.float32)
        qs = q_ck * scale
        ksq = k_ck * k_ck
        ibarK = tl.maximum(
            tl.load(Ibar + ((i_bh * nc + c) * C + o_c[:, None]) * D_K + o_k[None, :],
                    mask=mask_k[None, :], other=1.0).to(tl.float32), 1e-5)
        Ibar_S = tl.maximum(tl.sum(tl.where(mask_k[None, :], ibarK, 0.0), axis=1) / D_K, 1e-5)
        Qtil = qs / ibarK
        sc_raw = tl.dot(Qtil.to(tl.float32), tl.trans(k_ck).to(tl.float32))
        sc = sc_raw * Dm

        # ---- sweep A: Ibar_c_K (v-collapse), Ibar_c_S (vk) ----
        Ibar_c_K = tl.zeros([BK], dtype=tl.float32)
        sum_Ic = tl.zeros([1], dtype=tl.float32)
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (nc + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), 1e-5)
            Ibar_c_K += tl.sum(tl.where(mask_v[:, None], I_cv, 0.0), axis=0)
            sum_Ic += tl.sum(tl.where(mask_kv, I_cv, 0.0))
        Ibar_c_K = Ibar_c_K / D_V
        Ibar_c_S = tl.maximum(tl.sum(sum_Ic) / (D_V * D_K), 1e-5)
        Q_carry = qs * (Ibar_c_K[None, :] / ibarK)

        # ---- sweep B: carry dM/dI + dIbarV(carry+local) + Ibar_S/Ibar_c_* pieces ----
        dIbar_S = tl.zeros([C], dtype=tl.float32)
        dQtil_carry = tl.zeros([C, BK], dtype=tl.float32)
        dIbar_c_K_carry = tl.zeros([BK], dtype=tl.float32)
        dIbar_c_S_carry = tl.zeros([1], dtype=tl.float32)
        dsc = tl.zeros([C, C], dtype=tl.float32)
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (nc + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), 1e-5)
            mu_v = M_cv / I_cv
            do_cv = tl.load(do + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                            mask=mask_v[None, :], other=0.0).to(tl.float32)
            v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            ibarV = tl.maximum(
                tl.load(IbarV + ((i_bh * nc + c) * C + o_c[:, None]) * D_V + o_v[None, :],
                        mask=mask_v[None, :], other=1.0).to(tl.float32), 1e-5)
            Ibar_c_V = tl.sum(tl.where(mask_kv, I_cv, 0.0), axis=1) / D_K   # (BV,) per tile

            SrV = Ibar_S / Ibar_c_S
            rV = Ibar_c_V[None, :] / ibarV
            carry_ratio = rV * SrV[:, None]
            base_pre = tl.dot(Q_carry.to(tl.float32), tl.trans(mu_v).to(tl.float32))
            dbase_pre = do_cv * carry_ratio * cd[:, None]
            dcarry_ratio = do_cv * base_pre * cd[:, None]
            dRV = dcarry_ratio * SrV[:, None]
            dSrV = tl.sum(dcarry_ratio * rV, axis=1)
            dIbarV_carry = -dRV * Ibar_c_V[None, :] / (ibarV * ibarV)
            dIbar_S += dSrV / Ibar_c_S
            dIbar_c_S_carry += -tl.sum(dSrV * Ibar_S / (Ibar_c_S * Ibar_c_S))
            dmu = tl.dot(tl.trans(dbase_pre).to(tl.float32), Q_carry.to(tl.float32))
            dQ_carry = tl.dot(dbase_pre.to(tl.float32), mu_v.to(tl.float32))
            dQtil_carry += dQ_carry
            dIbar_c_K_carry += tl.sum(dQ_carry * qs / ibarK, axis=0)
            dM_cv = dmu / I_cv
            dI_cv_carry = -dmu * M_cv / (I_cv * I_cv)
            dIbar_c_V_carry = tl.sum(dRV / ibarV, axis=0)        # (BV,)

            # local path
            yl_pre = tl.dot(sc.to(tl.float32), v_cv.to(tl.float32))
            Smul = Ibar_S[:, None] / ibarV
            dyl_pre = do_cv * Smul
            dSmul = do_cv * yl_pre
            dIbar_S += tl.sum(dSmul / ibarV, axis=1)
            dIbarV_local = -dSmul * Ibar_S[:, None] / (ibarV * ibarV)
            dsc += tl.dot(dyl_pre.to(tl.float32), tl.trans(v_cv).to(tl.float32))

            # writes: dM_local(carry), dI_local_carry(carry + Ibar_c_V carry/DK), dIbarV, dv
            off_out = (i_bh * nc + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
            tl.store(dM_local_out + off_out, dM_cv, mask=mask_kv)
            dI_part = dI_cv_carry + tl.where(mask_v[:, None], dIbar_c_V_carry[:, None], 0.0) / D_K
            tl.store(dI_carry_out + off_out, dI_part, mask=mask_kv)
            off_ibv = ((i_bh * nc + c) * C + o_c[:, None]) * D_V + o_v[None, :]
            tl.store(dIbarV_out + off_ibv, dIbarV_carry + dIbarV_local, mask=mask_v[None, :])
            # dv[s,v] = sum_t score[t,s]*dyl_pre[t,v]
            dv_sv = tl.dot(tl.trans(sc).to(tl.float32), dyl_pre.to(tl.float32))  # (C_s,BV)
            tl.store(dv_out + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                     dv_sv, mask=mask_v[None, :])

        # ---- dIbar_K reverse path -> dq, dk_local, db_K (via dbetabar_K), dgt_K ----
        dscore_raw = dsc * Dm
        dQtil_local = tl.dot(dscore_raw.to(tl.float32), k_ck.to(tl.float32))      # (C,BK)
        dk_local = tl.dot(tl.trans(dscore_raw).to(tl.float32), Qtil.to(tl.float32))  # (C,BK)
        dIbar_K = -(dQtil_local * qs / (ibarK * ibarK)) \
                  - (dQtil_carry * qs * Ibar_c_K[None, :] / (ibarK * ibarK)) \
                  + dIbar_S[:, None] / D_K
        # the dIbar_S/D_K broadcast populates padded k-lanes; zero them so the
        # over-k reductions (dcd_fromK, dabar_K) don't leak padding.
        dIbar_K = tl.where(mask_k[None, :], dIbar_K, 0.0)
        dqs = (dQtil_local / ibarK) + (dQtil_carry * Ibar_c_K[None, :] / ibarK)
        dq_acc = dqs * scale

        # scan-K jacobian:  Ibar_K = Ip + cd*abar_K + Dm @ srcK ; srcK=betabar_K*ksq
        dabar_K = tl.sum(dIbar_K * cd[:, None], axis=0)            # (BK,) entry grad
        dcd_fromK = tl.sum(dIbar_K * (Ibar_c_K[None, :] - b_Ip), axis=1)  # (C,) abar_K=Ibar_c_K-Ip
        dsrcK = tl.dot(tl.trans(Dm).to(tl.float32), dIbar_K.to(tl.float32))  # (C,BK) [i,k]
        # dDm_K[t,i] = sum_k dIbar_K[t,k]*srcK[i,k] ; srcK[i,k]=betabar_K[i]*ksq[i,k]
        # need betabar_K (mean_v b). compute in a tiny sweep.
        betabar_K = tl.zeros([C], dtype=tl.float32)
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            betabar_K += tl.sum(tl.where(mask_v[None, :], b_cv, 0.0), axis=1)
        betabar_K = betabar_K / D_V
        srcK = betabar_K[:, None] * ksq                           # (C,BK) [i,k]
        dDm_K = tl.dot(dIbar_K.to(tl.float32), tl.trans(srcK).to(tl.float32))  # (C,C)[t,i]
        dbetabar_K = tl.sum(dsrcK * ksq, axis=1)                  # (C,)
        dksq_fromK = dsrcK * betabar_K[:, None]                   # (C,BK)

        # dclogf: local Dmask path + Dm_K path + cd path(carry+K)
        dDmask_local = dsc * sc_raw                               # only where tri (sc has Dm; sc_raw raw)
        # NOTE dscore in mirror: dscore=dyl_pre@v^T (=dsc); dscore_raw=dsc*Dm*tri;
        #   dDmask = dsc*sc_raw*tri  -> but sc_raw is raw score(Qtil@k^T). tri applied via Dm.
        dDmask_local = dsc * sc_raw * tl.where(tri, 1.0, 0.0)
        # dcd carry
        dcd_carry = tl.zeros([C], dtype=tl.float32)
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (nc + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), 1e-5)
            mu_v = M_cv / I_cv
            do_cv = tl.load(do + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                            mask=mask_v[None, :], other=0.0).to(tl.float32)
            ibarV = tl.maximum(
                tl.load(IbarV + ((i_bh * nc + c) * C + o_c[:, None]) * D_V + o_v[None, :],
                        mask=mask_v[None, :], other=1.0).to(tl.float32), 1e-5)
            Ibar_c_V = tl.sum(tl.where(mask_kv, I_cv, 0.0), axis=1) / D_K
            SrV = Ibar_S / Ibar_c_S
            carry_ratio = (Ibar_c_V[None, :] / ibarV) * SrV[:, None]
            base_pre = tl.dot(Q_carry.to(tl.float32), tl.trans(mu_v).to(tl.float32))
            dcd_carry += tl.sum(do_cv * base_pre * carry_ratio, axis=1)

        dclogf = tl.zeros([C], dtype=tl.float32)
        dDexp = dDmask_local * tl.exp(clogf[:, None] - clogf[None, :])
        dclogf += tl.sum(dDexp, axis=1)
        dclogf += -tl.sum(dDexp, axis=0)
        dDm_exp = dDm_K * Dm
        dclogf += tl.sum(dDm_exp, axis=1)
        dclogf += -tl.sum(dDm_exp, axis=0)
        dclogf += (dcd_carry + dcd_fromK) * cd

        total = tl.sum(dclogf)
        csum = tl.cumsum(dclogf, axis=0)
        dlogf = total - csum + dclogf
        dgt_K = dlogf * (-b_g)

        dk_K = dk_local + 2.0 * k_ck * dksq_fromK
        tl.store(dq_out + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :], dq_acc,
                 mask=mask_k[None, :])
        tl.store(dk_out + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :], dk_K,
                 mask=mask_k[None, :])
        tl.store(dgt_out + base_gt + o_c * H, dgt_K, mask=o_c < C)
        # db_K = dbetabar_K / D_V broadcast over v
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            db_K = tl.where(mask_v[None, :], dbetabar_K[:, None], 0.0) / D_V   # (C,BV)
            tl.store(db_out + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :], db_K,
                     mask=mask_v[None, :])

        # finalize: store the K-side dI contribution (Ibar_c_K carry + dabar_K)/D_V
        # and Ibar_c_S carry/(DV*DK) into its OWN buffer (pure store, no RMW).
        dIbar_c_K_total = dIbar_c_K_carry + dabar_K
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_out = (i_bh * nc + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
            add = tl.where(mask_v[:, None], dIbar_c_K_total[None, :], 0.0) / D_V \
                  + tl.where(mask_kv, tl.sum(dIbar_c_S_carry), 0.0) / (D_V * D_K)
            tl.store(dI_K_out + off_out, add, mask=mask_kv)

    @triton.autotune(configs=_fp_bwd_v_autotune(), key=['D_K', 'D_V', 'C'])
    @triton.jit
    def _fp_bwd_kernel_ibarv(
        k, b, gt, g, Ip, I_bound, dIbarV,
        dI_V_out, dk_V_out, db_V_out, dgt_V_out,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
    ):
        i_c, i_bh = tl.program_id(0), tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        nc = T // C
        c = i_c
        o_c = tl.arange(0, C)
        o_k = tl.arange(0, BK)
        mask_k = o_k < D_K
        NV = tl.cdiv(D_V, BV)

        b_Ip = tl.load(Ip + i_h).to(tl.float32)
        b_g = tl.load(g + i_h).to(tl.float32)
        base_qk = (i_b * T * H + c * C * H + i_h) * D_K
        base_vo = (i_b * T * H + c * C * H + i_h) * D_V
        base_gt = (i_b * T + c * C) * H + i_h

        gt_c = tl.load(gt + base_gt + o_c * H, mask=o_c < C, other=0.0).to(tl.float32)
        logf = -gt_c * b_g
        clogf = tl.cumsum(logf, axis=0)
        cd = tl.exp(clogf)
        tri = o_c[:, None] >= o_c[None, :]
        Dm = tl.where(tri, tl.exp(clogf[:, None] - clogf[None, :]), 0.0)

        k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_k[None, :], other=0.0).to(tl.float32)
        ksq = k_ck * k_ck
        ksqbar_V = tl.sum(tl.where(mask_k[None, :], ksq, 0.0), axis=1) / D_K

        dcd_fromV = tl.zeros([C], dtype=tl.float32)
        dDm_V = tl.zeros([C, C], dtype=tl.float32)
        dksqbar_V = tl.zeros([C], dtype=tl.float32)
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (nc + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), 1e-5)
            Ibar_c_V = tl.sum(tl.where(mask_kv, I_cv, 0.0), axis=1) / D_K
            abar_V = Ibar_c_V - b_Ip
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            dIbarV_cv = tl.load(dIbarV + ((i_bh * nc + c) * C + o_c[:, None]) * D_V + o_v[None, :],
                                mask=mask_v[None, :], other=0.0).to(tl.float32)
            srcV = ksqbar_V[:, None] * b_cv

            dabar_V = tl.sum(dIbarV_cv * cd[:, None], axis=0)                   # (BV,)
            dcd_fromV += tl.sum(dIbarV_cv * abar_V[None, :], axis=1)
            dsrcV = tl.dot(tl.trans(Dm).to(tl.float32), dIbarV_cv.to(tl.float32))   # (C,BV)[i]
            dDm_V += tl.dot(dIbarV_cv.to(tl.float32), tl.trans(srcV).to(tl.float32))
            db_V = dsrcV * ksqbar_V[:, None]
            dksqbar_V += tl.sum(dsrcV * b_cv, axis=1)

            tl.store(db_V_out + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :], db_V,
                     mask=mask_v[None, :])
            off_il = (i_bh * nc + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :]
            add = tl.where(mask_v[:, None], dabar_V[:, None], 0.0) / D_K
            tl.store(dI_V_out + off_il, add, mask=mask_kv)

        dclogf = tl.zeros([C], dtype=tl.float32)
        dDm_exp = dDm_V * Dm
        dclogf += tl.sum(dDm_exp, axis=1)
        dclogf += -tl.sum(dDm_exp, axis=0)
        dclogf += dcd_fromV * cd
        total = tl.sum(dclogf)
        csum = tl.cumsum(dclogf, axis=0)
        dlogf = total - csum + dclogf
        dgt_V = dlogf * (-b_g)
        tl.store(dgt_V_out + base_gt + o_c * H, dgt_V, mask=o_c < C)

        dksq_fromV = dksqbar_V[:, None] / D_K
        dk_V = 2.0 * k_ck * dksq_fromV
        tl.store(dk_V_out + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :], dk_V,
                 mask=mask_k[None, :])

    @triton.autotune(configs=_fp_bwd_autotune(), key=['D_K', 'D_V', 'C'])
    @triton.jit
    def _fp_bwd_kernel_scan(
        dM_local, dI_local, gt, g,
        dM_bound, dI_bound,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr
    ):
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

    @triton.autotune(configs=_fp_bwd_v_autotune(), key=['D_K', 'D_V', 'C'])
    @triton.jit
    def _fp_bwd_kernel_intra(
        k, v, b, gt, g, Ip,
        M_bound, I_bound,
        dM_bound, dI_bound,
        dk_b_out, dv_b_out, db_b_out, dgt_b_out,
        T, H: tl.constexpr,
        D_K: tl.constexpr, D_V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr,
    ):
        # boundary-update VJP only (V1-identical math, dpf = cM + cI)
        i_c, i_bh = tl.program_id(0), tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        nc = T // C
        c = i_c
        o_c = tl.arange(0, C)
        o_k = tl.arange(0, BK)
        mask_k = o_k < D_K
        NV = tl.cdiv(D_V, BV)

        b_Ip = tl.load(Ip + i_h).to(tl.float32)
        b_g = tl.load(g + i_h).to(tl.float32)
        base_qk = (i_b * T * H + c * C * H + i_h) * D_K
        base_vo = (i_b * T * H + c * C * H + i_h) * D_V
        base_gt = (i_b * T + c * C) * H + i_h

        k_ck = tl.load(k + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :],
                       mask=mask_k[None, :], other=0.0).to(tl.float32)
        gt_c = tl.load(gt + base_gt + o_c * H, mask=o_c < C, other=0.0).to(tl.float32)
        ksq = k_ck * k_ck
        logf = -gt_c * b_g
        clogf = tl.cumsum(logf, axis=0)
        clogf_last = tl.sum(tl.where(o_c == (C - 1), clogf, 0.0))
        pf = tl.exp(clogf_last)
        w = tl.exp(clogf_last - clogf)            # (C,)

        cM = tl.zeros([1], dtype=tl.float32)
        cI = tl.zeros([1], dtype=tl.float32)
        dw = tl.zeros([C], dtype=tl.float32)
        dk_b = tl.zeros([C, BK], dtype=tl.float32)
        for iv in range(NV):
            o_v = iv * BV + tl.arange(0, BV)
            mask_v = o_v < D_V
            mask_kv = mask_v[:, None] & mask_k[None, :]
            off_st = ((i_bh * (nc + 1) + c) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            M_cv = tl.load(M_bound + off_st, mask=mask_kv, other=0.0).to(tl.float32)
            I_cv = tl.maximum(tl.load(I_bound + off_st, mask=mask_kv, other=1.0).to(tl.float32), 1e-5)
            off_dn = ((i_bh * (nc + 1) + c + 1) * D_V * D_K + o_v[:, None] * D_K + o_k[None, :])
            dM_v = tl.load(dM_bound + off_dn, mask=mask_kv, other=0.0).to(tl.float32)
            dI_v = tl.load(dI_bound + off_dn, mask=mask_kv, other=0.0).to(tl.float32)
            v_cv = tl.load(v + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            b_cv = tl.load(b + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :],
                           mask=mask_v[None, :], other=0.0).to(tl.float32)
            vdM = tl.dot(v_cv.to(tl.float32), dM_v.to(tl.float32))   # (C,BK)
            bdI = tl.dot(b_cv.to(tl.float32), dI_v.to(tl.float32))   # (C,BK)
            # dv_b = w * (k . dM_next) ; dv[t,v]=w_t*sum_k k[t,k]*dM_next[v,k]
            dv_b = w[:, None] * tl.dot(k_ck.to(tl.float32), tl.trans(dM_v).to(tl.float32))  # (C,BV)
            tl.store(dv_b_out + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :], dv_b,
                     mask=mask_v[None, :])
            # db_b = w * (ksq . dI_next) ; (C,BV)
            db_b = w[:, None] * tl.dot(ksq.to(tl.float32), tl.trans(dI_v).to(tl.float32))
            tl.store(db_b_out + base_vo + o_c[:, None] * (H * D_V) + o_v[None, :], db_b,
                     mask=mask_v[None, :])
            cM += tl.sum(M_cv * dM_v)
            cI += tl.sum((I_cv - b_Ip) * dI_v)
            dw += tl.sum(vdM * k_ck, axis=1) + tl.sum(bdI * ksq, axis=1)
            dk_b += w[:, None] * (vdM + 2.0 * k_ck * bdI)
        tl.store(dk_b_out + base_qk + o_c[:, None] * (H * D_K) + o_k[None, :], dk_b,
                 mask=mask_k[None, :])

        dpf = tl.sum(cM) + tl.sum(cI)
        dclast = dpf * pf + tl.sum(dw * w)
        dclogf_b = -dw * w
        dclogf_b = dclogf_b + tl.where(o_c == (C - 1), dclast, 0.0)
        total = tl.sum(dclogf_b)
        csum = tl.cumsum(dclogf_b, axis=0)
        dlogf = total - csum + dclogf_b
        dgt_b = dlogf * (-b_g)
        tl.store(dgt_b_out + base_gt + o_c * H, dgt_b, mask=o_c < C)

    def _fp_clamp_C(D_K, D_V, C, dev):
        return C

    def _fast_palimpsa_bwd_triton(do, q, k, v, b, gt, g, Ip, scale, C,
                                  M_bound, I_bound, Ibar, IbarV):
        B, T, H, D_K = q.shape
        D_V = v.shape[-1]
        nc = T // C
        dev = q.device
        BK_full = triton.next_power_of_2(D_K)
        assert (not torch.is_tensor(Ip)) or Ip.numel() == H, "scalar Ip only (V2)"

        Ipf = (Ip.float().contiguous() if torch.is_tensor(Ip)
               else torch.full((H,), float(Ip), device=dev, dtype=torch.float32))
        qc, kc, vc, bc, gtc = [x.contiguous() for x in (q, k, v, b, gt)]
        gf = g.float().contiguous()
        dof = do.contiguous().float()

        f32 = torch.float32
        dM_local = torch.zeros(B * H, nc, D_V, D_K, device=dev, dtype=f32)
        dI_carry = torch.zeros(B * H, nc, D_V, D_K, device=dev, dtype=f32)
        dI_K = torch.zeros(B * H, nc, D_V, D_K, device=dev, dtype=f32)
        dI_V = torch.zeros(B * H, nc, D_V, D_K, device=dev, dtype=f32)
        dM_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=f32)
        dI_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=f32)
        dIbarV = torch.zeros(B * H, nc, C, D_V, device=dev, dtype=f32)

        # split output buffers (host-reduced; no atomics, no in-kernel RMW)
        dq_r = torch.zeros(B, T, H, D_K, device=dev, dtype=f32)
        dk_r = torch.zeros(B, T, H, D_K, device=dev, dtype=f32)
        dv_r = torch.zeros(B, T, H, D_V, device=dev, dtype=f32)
        db_r = torch.zeros(B, T, H, D_V, device=dev, dtype=f32)
        dgt_r = torch.zeros(B, T, H, device=dev, dtype=f32)
        dk_V = torch.zeros(B, T, H, D_K, device=dev, dtype=f32)
        db_V = torch.zeros(B, T, H, D_V, device=dev, dtype=f32)
        dgt_V = torch.zeros(B, T, H, device=dev, dtype=f32)
        dk_b = torch.zeros(B, T, H, D_K, device=dev, dtype=f32)
        dv_b = torch.zeros(B, T, H, D_V, device=dev, dtype=f32)
        db_b = torch.zeros(B, T, H, D_V, device=dev, dtype=f32)
        dgt_b = torch.zeros(B, T, H, device=dev, dtype=f32)

        # pass 1: full read VJP
        grid_c = lambda META: (nc, B * H)
        _fp_bwd_kernel_local_state[grid_c](
            qc, kc, vc, bc, gtc, gf, Ipf,
            M_bound, I_bound, Ibar, IbarV, dof,
            dM_local, dI_carry, dI_K, dIbarV,
            dq_r, dk_r, dv_r, db_r, dgt_r, scale,
            T, H=H, D_K=D_K, D_V=D_V, BK=BK_full, C=C)

        # pass 1.5: ibarv scan (writes dI_V, dk_V, db_V, dgt_V)
        _fp_bwd_kernel_ibarv[grid_c](
            kc, bc, gtc, gf, Ipf, I_bound, dIbarV,
            dI_V, dk_V, db_V, dgt_V,
            T, H=H, D_K=D_K, D_V=D_V, BK=BK_full, C=C)

        # combine the three dI contributions BEFORE the scan (host add, no RMW)
        dI_local = dI_carry + dI_K + dI_V

        # pass 2: scan (BK/BV tiled, V1-identical)
        grid_sc = lambda META: (triton.cdiv(D_K, META['BK']), triton.cdiv(D_V, META['BV']), B * H)
        _fp_bwd_kernel_scan[grid_sc](
            dM_local, dI_local, gtc, gf, dM_bound, dI_bound,
            T, H=H, D_K=D_K, D_V=D_V, C=C)

        # pass 3: boundary VJP
        _fp_bwd_kernel_intra[grid_c](
            kc, vc, bc, gtc, gf, Ipf,
            M_bound, I_bound, dM_bound, dI_bound,
            dk_b, dv_b, db_b, dgt_b,
            T, H=H, D_K=D_K, D_V=D_V, BK=BK_full, C=C)

        dq = dq_r
        dk = dk_r + dk_V + dk_b
        dv = dv_r + dv_b
        db = db_r + db_V + db_b
        dgt = dgt_r + dgt_V + dgt_b
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
            assert obv >= D_V, "V2 output kernel needs BV>=D_V (full value residency)."
            if not torch.is_tensor(Ip):
                Ip = torch.full((H,), float(Ip), device=dev, dtype=torch.float32)
            assert Ip.numel() == H, "fast_palimpsa_v2 supports scalar-per-head Ip (H,) only."
            Ipf = Ip.float().contiguous()
            gf = g.float().contiguous()
            M_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)
            I_bound = torch.empty(B * H, nc + 1, D_V, D_K, device=dev, dtype=torch.float32)
            Ibar = torch.empty(B * H, nc, C, D_K, device=dev, dtype=torch.float32)
            IbarV = torch.empty(B * H, nc, C, D_V, device=dev, dtype=torch.float32)
            o = torch.empty(B, T, H, D_V, device=dev, dtype=q.dtype)
            qc, kc, vc, bc, gtc = [x.contiguous() for x in (q, k, v, b, gt)]

            _fp_state_kernel[(triton.cdiv(D_K, BK), triton.cdiv(D_V, BV), B * H)](
                kc, vc, bc, gtc, gf, Ipf, M_bound, I_bound,
                T, H=H, D_K=D_K, D_V=D_V, BK=BK, BV=BV, C=C, PERDK=False)

            _fp_ibar_kernel[(triton.cdiv(D_K, BK), B * H)](
                kc, bc, gtc, gf, Ipf, I_bound, Ibar, IbarV,
                T, H=H, D_K=D_K, D_V=D_V, BK=BK, C=C,
                BV_IB=triton.next_power_of_2(D_V),
                BK_IB=triton.next_power_of_2(D_K))

            _fp_output_kernel[(triton.cdiv(D_V, obv), nc, B * H)](
                qc, kc, vc, gtc, gf, M_bound, I_bound, Ibar, IbarV, o, scale,
                T, H=H, D_K=D_K, D_V=D_V, BK=obk, BV=obv, C=C,
                num_warps=onw, num_stages=ons)

            ctx.save_for_backward(q, k, v, b, gt, gf, Ipf, M_bound, I_bound, Ibar, IbarV)
            ctx.scale, ctx.C = scale, C
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, b, gt, g, Ip, M_bound, I_bound, Ibar, IbarV = ctx.saved_tensors
            dq, dk, dv, db, dgt = _fast_palimpsa_bwd_triton(
                do, q, k, v, b, gt, g, Ip, ctx.scale, ctx.C,
                M_bound, I_bound, Ibar, IbarV)
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
    """CPU-only: exercises the reference across odd D_K/D_V (scalar Ip only)."""
    for (DK, DV) in [(128, 16), (48, 40), (33, 7), (64, 128), (96, 192)]:
        torch.manual_seed(0)
        B, L, H, C = 1, 32, 2, 16
        q = torch.randn(B, L, H, DK, dtype=torch.float64)
        k = torch.nn.functional.normalize(torch.randn(B, L, H, DK, dtype=torch.float64), dim=-1)
        v = torch.randn(B, L, H, DV, dtype=torch.float64)
        b = torch.rand(B, L, H, DV, dtype=torch.float64) * 1.5 + 0.1
        gt = torch.rand(B, L, H, dtype=torch.float64) * 0.1
        g = torch.rand(H, dtype=torch.float64) * 0.5 + 0.5
        Ip = torch.rand(H, dtype=torch.float64) + 0.5      # scalar per head ONLY
        scale = DK ** -0.5
        y = fast_palimpsa_ref(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=C)
        assert y.shape == (B, L, H, DV)
        # TRUE invariant for the rank-1 frozen-carry mode: with NO carried history
        # (a single chunk, L == C) AND beta constant across D_V (so the V-collapse
        # is lossless), the rank-1 approximation is EXACT. (With history or
        # anisotropic beta it is genuinely approximate -- that is the method.)
        qs, ks, vs, gts = q[:, :C], k[:, :C], v[:, :C], gt[:, :C]
        bs = b[:, :C].mean(-1, keepdim=True).expand(B, C, H, DV).contiguous()
        ye = _exact_recurrence(qs, ks, vs, bs, gts, g, Ip, scale)
        ya = fast_palimpsa_ref(qs, ks, vs, bs, gts, g, Ip, scale=scale, chunk_size=C)
        err = (ya - ye).abs().max().item()
        print(f"DK={DK:3d} DV={DV:3d} scalar-Ip: shape ok, single-chunk const-beta exactness {err:.2e}")
        assert err < 1e-9, f"single-chunk const-beta should be exact, got {err}"

    # vec training path vs loop reference: fwd + all grads to fp64 precision
    torch.manual_seed(1)
    B, L, H, C, DK, DV = 2, 64, 3, 16, 48, 96
    mk = lambda *s: torch.randn(*s, dtype=torch.float64)
    q = mk(B, L, H, DK).requires_grad_(True)
    k = torch.nn.functional.normalize(mk(B, L, H, DK), dim=-1).detach().requires_grad_(True)
    v = mk(B, L, H, DV).requires_grad_(True)
    b = (torch.rand(B, L, H, DV, dtype=torch.float64) * 1.5 + 0.1).requires_grad_(True)
    gt = (torch.rand(B, L, H, dtype=torch.float64) * 0.1).requires_grad_(True)
    g = torch.rand(H, dtype=torch.float64) * 0.5 + 0.5
    Ip = torch.rand(H, dtype=torch.float64) + 0.5
    scale = DK ** -0.5
    yv = fast_palimpsa_vec(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=C)
    yr = fast_palimpsa_ref(q.detach(), k.detach(), v.detach(), b.detach(),
                           gt.detach(), g, Ip, scale=scale, chunk_size=C)
    do = torch.randn_like(yv)
    gv = torch.autograd.grad(yv, (q, k, v, b, gt), do)
    q2, k2, v2, b2, gt2 = [t.detach().clone().requires_grad_(True) for t in (q, k, v, b, gt)]
    yr2 = fast_palimpsa_ref(q2, k2, v2, b2, gt2, g, Ip, scale=scale, chunk_size=C)
    gr = torch.autograd.grad(yr2, (q2, k2, v2, b2, gt2), do)
    fwd = (yv - yr).abs().max().item()
    worst = max(((a - bb).abs().max() / (bb.abs().max() + 1e-30)).item() for a, bb in zip(gv, gr))
    print(f"vec vs loop-ref: fwd {fwd:.2e}, worst grad rel {worst:.2e}")
    assert fwd < 1e-9 and worst < 1e-9


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