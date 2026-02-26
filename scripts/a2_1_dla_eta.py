"""DLA with different η exponents (Assignment 2.1.A).

Runs PDE-based DLA for η ∈ {0, 0.5, 1, 2, 3} and shows how the cluster
morphology changes.  η<1 → more compact (Eden-like); η>1 → more branched.

Performance note:
    High η concentrates growth at exposed tips → very thin, branched clusters.
    Each tip addition significantly perturbs the local concentration field, so
    the warm-start SOR advantage diminishes and more BVP iterations are needed
    per step.  We therefore use a smaller max_iter cap and fewer steps for
    high η, which is also an illustration of the computational trade-off.
"""

import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots  # noqa: F401

styles = ["science"] if (shutil.which("latex") and shutil.which("dvipng")) else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.models.dla import run_dla
from scicomp3.bvp.omega import get_optimal_omega

# ── Parameters ──────────────────────────────────────────────────────────────
N     = 100
SEED  = 7
OMEGA = get_optimal_omega(N)

# High-η clusters grow as thin filaments; reduce step count to keep runtime
# reasonable.  The morphology contrast is visible even with fewer steps.
ETA_CONFIGS = [
    (0,   150, 500),   # (eta, n_steps, max_iter_per_step)
    (0.5, 150, 500),
    (1.0, 200, 1000),
    (2.0, 150, 500),
    (3.0, 100, 300),
]

x_grid = np.linspace(0, 1, N + 1)
y_grid = np.linspace(0, 1, N + 1)
X, Y   = np.meshgrid(x_grid, y_grid, indexing="ij")

fig, axes = plt.subplots(1, len(ETA_CONFIGS), figsize=(4 * len(ETA_CONFIGS), 5))

for ax, (eta, n_steps, max_iter) in zip(axes, ETA_CONFIGS):
    print(f"Running DLA  η={eta:4.1f}  steps={n_steps} …", end=" ", flush=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    cluster_mask, concentration, n_iters = run_dla(
        N=N, n_steps=n_steps, eta=eta, omega=OMEGA, max_iter=max_iter, rng=rng
    )
    elapsed = time.perf_counter() - t0
    n_cluster = cluster_mask.sum()
    mean_iters = np.mean(n_iters)
    print(f"done  ({n_cluster} sites, {elapsed:.1f}s, mean SOR iters={mean_iters:.0f})")

    display = np.where(cluster_mask, 1.0, np.nan)
    ax.pcolormesh(X, Y, display, shading="nearest", cmap="Reds", vmin=0, vmax=1)
    ax.set_title(
        f"$\\eta = {eta}$\n"
        f"{n_cluster} sites,  $\\bar{{k}}_{{\\mathrm{{SOR}}}}={mean_iters:.0f}$"
    )
    ax.set_xlabel("$x$")
    if ax is axes[0]:
        ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

fig.suptitle(
    f"DLA cluster morphology for different $\\eta$  (N={N},  $\\omega={OMEGA:.3f}$)\n"
    r"$\eta<1$: compact (Eden-like) $\quad$ "
    r"$\eta=1$: standard DLA $\quad$ "
    r"$\eta>1$: branched (lightning-like)"
)
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / "a2_1_dla_eta.png"
plt.savefig(fname, dpi=150)
print(f"Saved → {fname}")
plt.show()
