"""MC DLA with different sticking probabilities (Assignment 2.2.D).

Demonstrates how the sticking probability ``ps`` controls cluster morphology:
- ps=1.0  →  standard DLA (fractal, open clusters)
- ps<1.0  →  walkers survive more collisions → more compact, Eden-like clusters
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots  # noqa: F401

styles = ["science"] if (shutil.which("latex") and shutil.which("dvipng")) else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.models.mc_dla import run_mc_dla

# ── Parameters ──────────────────────────────────────────────────────────────
N         = 100
N_STEPS   = 200
PS_VALUES = [1.0, 0.5, 0.1, 0.01]
SEED      = 13

x_grid = np.linspace(0, 1, N + 1)
y_grid = np.linspace(0, 1, N + 1)
X, Y   = np.meshgrid(x_grid, y_grid, indexing="ij")

fig, axes = plt.subplots(1, len(PS_VALUES), figsize=(4 * len(PS_VALUES), 4.5))

for ax, ps in zip(axes, PS_VALUES):
    print(f"Running MC DLA  ps={ps} …", end=" ", flush=True)
    rng = np.random.default_rng(SEED)
    cluster_mask, walk_lengths = run_mc_dla(N=N, n_steps=N_STEPS, ps=ps, rng=rng)
    n_cluster = cluster_mask.sum()
    mean_walk = np.mean(walk_lengths)
    print(f"done  ({n_cluster} sites, mean walk={mean_walk:.0f})")

    display = np.where(cluster_mask, 1.0, np.nan)
    ax.pcolormesh(X, Y, display, shading="nearest", cmap="Greens", vmin=0, vmax=1)
    ax.set_title(f"$p_s = {ps}$\n({n_cluster} sites)")
    ax.set_xlabel("$x$")
    if ax is axes[0]:
        ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

fig.suptitle(f"MC DLA cluster morphology for different sticking probabilities $p_s$\n"
             f"(N={N}, {N_STEPS} particles)")
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / "a2_2_mc_dla_ps.png"
plt.savefig(fname, dpi=150)
print(f"Saved → {fname}")
plt.show()
