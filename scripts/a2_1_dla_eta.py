"""DLA with different η exponents (Assignment 2.1.A).

Runs PDE-based DLA for η ∈ {0, 0.5, 1, 2, 3} and shows how the cluster
morphology changes.  η<1 → more compact (Eden-like); η>1 → more branched.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots  # noqa: F401

styles = ["science"] if (shutil.which("latex") and shutil.which("dvipng")) else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.models.dla import run_dla
from scicomp3.bvp.omega import get_optimal_omega

# ── Parameters ──────────────────────────────────────────────────────────────
N = 100
N_STEPS = 250
ETA_VALUES = [0, 0.5, 1.0, 2.0, 3.0]
SEED = 7
OMEGA = get_optimal_omega(N)

x_grid = np.linspace(0, 1, N + 1)
y_grid = np.linspace(0, 1, N + 1)
X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

fig, axes = plt.subplots(1, len(ETA_VALUES), figsize=(4 * len(ETA_VALUES), 4.5))

for ax, eta in zip(axes, ETA_VALUES):
    print(f"Running DLA  η={eta} …", end=" ", flush=True)
    rng = np.random.default_rng(SEED)
    cluster_mask, concentration, n_iters = run_dla(
        N=N, n_steps=N_STEPS, eta=eta, omega=OMEGA, rng=rng
    )
    n_cluster = cluster_mask.sum()
    print(f"done  ({n_cluster} sites)")

    # Simple visualisation: binary cluster
    display = np.where(cluster_mask, 1.0, np.nan)
    ax.pcolormesh(X, Y, display, shading="nearest", cmap="Reds", vmin=0, vmax=1)
    ax.set_title(f"$\\eta = {eta}$\n({n_cluster} sites)")
    ax.set_xlabel("$x$")
    if ax is axes[0]:
        ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

fig.suptitle(f"DLA cluster morphology for different $\\eta$ (N={N}, {N_STEPS} steps)")
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / "a2_1_dla_eta.png"
plt.savefig(fname, dpi=150)
print(f"Saved → {fname}")
plt.show()
