# -*- coding: utf-8 -*-
# plot_fast_palimpsa_sawtooth.py
# Sawtooth error visualization for Fast Palimpsa. Place next to chunk_fast_palimpsa.py.
#
# Produces, for the output error and predictive-variance error:
#   * a 2D heatmap (x = token position over many chunks, y = forgetting rate g)
#   * a "sawtooth envelope" (mean error by within-chunk position) 
# Evaluated as Relative Error Percentage (%).

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chunk_fast_palimpsa import _exact_recurrence
from chunk_fast_palimpsa import fast_palimpsa_v3_ref as fast_palimpsa_ref


def make_plot(out_png="fast_palimpsa_sawtooth.png",
              B=1, H=1, D_K=16, D_V=16, C=16, n_chunks=12, n_rates=40,
              g_lo=0.05, g_hi=3.0, seed=2):
    torch.manual_seed(seed)
    L = C * n_chunks
    rates = torch.linspace(g_lo, g_hi, n_rates, dtype=torch.float64)

    q = torch.randn(B, L, H, D_K, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn(B, L, H, D_K, dtype=torch.float64), dim=-1)
    v = torch.randn(B, L, H, D_V, dtype=torch.float64)
    b = torch.rand(B, L, H, D_V, dtype=torch.float64) * 1.5 + 0.1
    gt = torch.rand(B, L, H, dtype=torch.float64) * 0.1
    Ip = torch.ones(H, dtype=torch.float64)
    scale = D_K ** -0.5

    Y_err = np.zeros((n_rates, L))
    V_err = np.zeros((n_rates, L))
    eps = 1e-9  # Prevent division by zero

    for r, rate in enumerate(rates):
        g = torch.full((H,), float(rate), dtype=torch.float64)
        ye, ve = _exact_recurrence(q, k, v, b, gt, g, Ip, scale)
        ya, va = fast_palimpsa_ref(q, k, v, b, gt, g, Ip, scale=scale,
                                   chunk_size=C, output_uncertainty=True)
        
        # Calculate Relative Error Percentage
        # y_rel = (ya - ye).abs() / (ye.abs() + eps) * 100.0
        # v_rel = (va - ve).abs() / (ve.abs() + eps) * 100.0

        y_rel = (ya - ye).abs() / (ye.abs().mean() + eps) * 100.0
        v_rel = (va - ve).abs() / (ve.abs().mean() + eps) * 100.0

        Y_err[r] = y_rel.max(-1).values[0, :, 0].numpy()
        V_err[r] = v_rel.max(-1).values[0, :, 0].numpy()

    nc = L // C
    fig, axes = plt.subplots(2, 2, figsize=(16, 8),
                             gridspec_kw={"width_ratios": [3, 1]})
    for row, data, name in [(0, Y_err, "Relative Output error (%)"),
                            (1, V_err, "Relative Variance error (%)")]:
        ax = axes[row, 0]
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="magma",
                       extent=[0, L, float(rates[0]), float(rates[-1])])
        for c in range(0, L + 1, C):
            ax.axvline(c, color="cyan", lw=0.4, alpha=0.5)
        ax.set_ylabel("forgetting rate g")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.025)

        env = data.reshape(n_rates, nc, C).mean(axis=(0, 1))
        ax2 = axes[row, 1]
        ax2.plot(np.arange(C), env, marker="o", color="crimson")
        ax2.axvline(0, color="cyan", lw=1.0, alpha=0.7)
        ax2.set_title("sawtooth envelope")
        ax2.set_xlabel("pos within chunk (0=boundary)")
        ax2.grid(alpha=0.3)
        axes[1, 0].set_xlabel(f"token position (cyan = chunk boundaries, every {C})")
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        print(f"saved {out_png}")

        bnd = Y_err[:, ::C].mean()
        mid = Y_err[:, C // 2::C].mean()
        print(f"[sawtooth] mean relative Y err boundary={bnd:.2f}% mid-chunk={mid:.2f}% ratio={mid/max(bnd,1e-30):.2f}x")

        n_points=10000
        # Flatten everything: (B*L*H*DV)
        ye_flat = ye.detach().cpu().flatten().numpy()
        ya_flat = ya.detach().cpu().flatten().numpy()
        
        # Subsample 1k points
        indices = np.random.choice(len(ye_flat), n_points, replace=False)
        y_true = ye_flat[indices]
        y_approx = ya_flat[indices]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_approx, alpha=0.3, s=10, label="Tokens")
        
        # Identity line
        min_val = min(y_true.min(), y_approx.min())
        max_val = max(y_true.max(), y_approx.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='x=y')
        
        plt.xlabel("Exact Palimpsa Output")
        plt.ylabel("Fast Palimpsa V3 Output")
        plt.title("Approximation Accuracy (10k Sample)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("palimpsa_scatter_V3.png")
        print("Scatter plot saved to palimpsa_scatter.png")


if __name__ == "__main__":
    make_plot("fast_palimpsa_sawtooth_DV64.png", D_K=32, D_V=64)