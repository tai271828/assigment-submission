"""DLA simulation using Red-Black SOR (vectorised NumPy).

Same DLA as a2_1_dla.py but uses the Red-Black SOR variant which
updates even/odd checkerboard sites in two vectorised half-sweeps
instead of a Python element loop.
"""

import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots  # noqa: F401

styles = (
    ["science"]
    if (shutil.which("latex") and shutil.which("dvipng"))
    else ["science", "no-latex"]
)
plt.style.use(styles)

from scicomp3.models.dla import run_dla
from scicomp3.bvp.omega import get_optimal_omega

# ── Parameters ──────────────────────────────────────────────────────────────
N = 100
N_STEPS = 300
ETA = 1.0
SEED = 42
OMEGA = get_optimal_omega(N)
METHOD = "sor_redblack"

print(f"DLA (Red-Black SOR): N={N}, n_steps={N_STEPS}, η={ETA}, ω={OMEGA:.4f}")

# ── Run simulation ───────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
t0 = time.perf_counter()
cluster_mask, concentration, n_iters = run_dla(
    N=N,
    n_steps=N_STEPS,
    eta=ETA,
    omega=OMEGA,
    rng=rng,
    method=METHOD,
)
elapsed = time.perf_counter() - t0

n_cluster = cluster_mask.sum()
print(f"Cluster size: {n_cluster} sites")
print(f"BVP iters per step — mean: {np.mean(n_iters):.1f}, max: {np.max(n_iters)}")
print(f"Wall time: {elapsed:.1f} s")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
x_grid = np.linspace(0, 1, N + 1)
y_grid = np.linspace(0, 1, N + 1)
X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

# 1. Cluster
ax = axes[0]
cluster_y = np.zeros_like(cluster_mask, dtype=float)
cluster_y[cluster_mask] = np.argwhere(cluster_mask)[:, 1] / N
cluster_display = np.where(cluster_mask, cluster_y, np.nan)
im0 = ax.pcolormesh(
    X, Y, cluster_display, shading="nearest", cmap="plasma", vmin=0, vmax=1
)
fig.colorbar(im0, ax=ax, label="Normalised y-position", fraction=0.046, pad=0.04)
ax.set_title(f"DLA cluster ($\\eta={ETA}$, {n_cluster} sites)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_aspect("equal")

# 2. Concentration field
ax = axes[1]
im1 = ax.pcolormesh(
    X, Y, concentration, shading="nearest", cmap="gist_heat", vmin=0, vmax=1
)
fig.colorbar(im1, ax=ax, label="$c(x,y)$", fraction=0.046, pad=0.04)
ax.set_title("Final concentration field")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_aspect("equal")

# 3. Iterations per step
ax = axes[2]
ax.plot(n_iters, lw=0.8)
ax.set_xlabel("Growth step")
ax.set_ylabel("SOR iterations")
ax.set_title("Red-Black SOR iterations per step\n(warm-start)")
ax.grid(True, alpha=0.3)

fig.suptitle(f"PDE-based DLA — Red-Black SOR ({N}×{N}, {elapsed:.1f} s)")
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / "a2_1_dla_redblack.png"
plt.savefig(fname, dpi=150)
print(f"Saved → {fname}")
plt.show()
