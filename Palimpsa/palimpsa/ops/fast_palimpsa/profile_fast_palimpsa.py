# profile_fast_palimpsa.py
# Put this next to chunk_fast_palimpsa.py and run:  python profile_fast_palimpsa.py
#
# It isolates the fast_palimpsa op (NOT the whole model) at your real training
# shape, warms up (so JIT/autotune compile is not counted), then times and
# profiles forward and backward separately. Tells us whether time is in the
# fwd state kernel, the 3 bwd kernels, or the Python split-buffer reductions.

import torch

# ---- import the op from your module ----------------------------------------
# Adjust the module name if your file is named differently.
from chunk_fast_palimpsa import chunk_fast_palimpsa, CHUNK_C

assert torch.cuda.is_available(), "need a GPU"
dev = "cuda"
dtype = torch.bfloat16          # match training

# ---- real per-head shape from your config ----------------------------------
# config: num_heads=16, head_dim=96 (D_K), expand_v=2 -> D_V=192, seq_len=4096,
# per-device batch_size=2. One forward call sees (B, T, H, D).
B, T, H, D_K, D_V = 2, 4096, 16, 96, 192
chunk_size = CHUNK_C            # whatever the model passes; clamp logic may change it

torch.manual_seed(0)
mk = lambda *s: torch.randn(*s, device=dev, dtype=dtype)
def fresh():
    q  = mk(B, T, H, D_K).requires_grad_(True)
    k  = torch.nn.functional.normalize(mk(B, T, H, D_K), dim=-1).detach().requires_grad_(True)
    v  = mk(B, T, H, D_V).requires_grad_(True)
    b  = (torch.rand(B, T, H, D_V, device=dev, dtype=dtype) * 1.5 + 0.1).requires_grad_(True)
    gt = (torch.rand(B, T, H, device=dev, dtype=dtype) * 0.1).requires_grad_(True)
    g  = torch.rand(H, device=dev, dtype=dtype) * 0.5 + 0.5
    Ip = torch.rand(H, device=dev, dtype=dtype) + 0.5
    return q, k, v, b, gt, g, Ip

scale = D_K ** -0.5

# ---- check which chunk size actually gets used (recompile-churn detector) ---
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    q, k, v, b, gt, g, Ip = fresh()
    o = chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=chunk_size)
    for wi in w:
        print("WARN:", wi.message)
print(f"requested chunk_size={chunk_size}; output shape={tuple(o.shape)}")

# ---- warmup (compile + autotune, NOT timed) --------------------------------
print("warming up (compiling kernels)...")
for _ in range(3):
    q, k, v, b, gt, g, Ip = fresh()
    o = chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=chunk_size)
    do = torch.randn_like(o)
    torch.autograd.grad(o, (q, k, v, b, gt), do)
torch.cuda.synchronize()

# ---- steady-state timing (separate fwd / bwd) ------------------------------
def time_it(fn, iters=20):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters    # ms per iter

# forward only
def fwd_only():
    q, k, v, b, gt, g, Ip = fresh()
    with torch.no_grad():
        chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=chunk_size)

# build a graph once, time backward repeatedly
q, k, v, b, gt, g, Ip = fresh()
o = chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=chunk_size)
do = torch.randn_like(o)
def bwd_only():
    torch.autograd.grad(o, (q, k, v, b, gt), do, retain_graph=True)

# full fwd+bwd
def full():
    q, k, v, b, gt, g, Ip = fresh()
    o = chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=chunk_size)
    do = torch.randn_like(o)
    torch.autograd.grad(o, (q, k, v, b, gt), do)

t_fwd  = time_it(fwd_only)
t_bwd  = time_it(bwd_only)
t_full = time_it(full)
print(f"\n[timing per call, B={B} T={T} H={H} D_K={D_K} D_V={D_V}]")
print(f"  forward only : {t_fwd:8.2f} ms")
print(f"  backward only: {t_bwd:8.2f} ms")
print(f"  fwd+bwd      : {t_full:8.2f} ms")

# ---- op-level profile of one full fwd+bwd ----------------------------------
print("\n== CUDA op profile (one fwd+bwd) ==")
q, k, v, b, gt, g, Ip = fresh()
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    o = chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=chunk_size)
    do = torch.randn_like(o)
    torch.autograd.grad(o, (q, k, v, b, gt), do)
torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25))

# ---- peak memory --------------------------------------------------------------
torch.cuda.reset_peak_memory_stats()
q, k, v, b, gt, g, Ip = fresh()
o = chunk_fast_palimpsa(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=chunk_size)
do = torch.randn_like(o)
torch.autograd.grad(o, (q, k, v, b, gt), do)
torch.cuda.synchronize()
print(f"\npeak CUDA mem for one fast_palimpsa fwd+bwd: "
      f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")
