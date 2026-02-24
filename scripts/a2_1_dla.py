"""DLA simulation using the Laplace equation (Assignment 2.1.A).

Runs a single PDE-based DLA growth simulation on a 100×100 grid with the
default exponent η=1 and plots:
  1. The grown cluster coloured by growth order.
  2. The final concentration field.
  3. Convergence: BVP iteration count per growth step.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

import scienceplots  # noqa: F401

styles = ["science"] if (shutil.which("latex") and shutil.which("dvipng")) else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.models.dla import run_dla, find_growth_candidates
from scicomp3.bvp.omega import get_optimal_omega

# ── Parameters ──────────────────────────────────────────────────────────────
N = 100
N_STEPS = 300
ETA = 1.0
SEED = 42
OMEGA = get_optimal_omega(N)

print(f"DLA: N={N}, n_steps={N_STEPS}, η={ETA}, ω={OMEGA:.4f}")

# ── Run simulation ───────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
cluster_mask, concentration, n_iters = run_dla(
    N=N,
    n_steps=N_STEPS,
    eta=ETA,
    omega=OMEGA,
    rng=rng,
)

n_cluster = cluster_mask.sum()
print(f"Cluster size: {n_cluster} sites")
print(f"BVP iters per step — mean: {np.mean(n_iters):.1f}, max: {np.max(n_iters)}")

# ── Build growth-order array for colouring ───────────────────────────────────
# Re-run growth order tracking via argwhere on the final mask.
# (We colour by approximate y-position as a proxy for growth order.)
cluster_y = np.zeros_like(cluster_mask, dtype=float)
cluster_y[cluster_mask] = np.argwhere(cluster_mask)[:, 1] / N

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
x_grid = np.linspace(0, 1, N + 1)
y_grid = np.linspace(0, 1, N + 1)
X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

# 1. Cluster coloured by y-position (proxy for growth depth)
ax = axes[0]
cluster_display = np.where(cluster_mask, cluster_y, np.nan)
im0 = ax.pcolormesh(X, Y, cluster_display, shading="nearest",
                    cmap="plasma", vmin=0, vmax=1)
fig.colorbar(im0, ax=ax, label="Normalised y-position", fraction=0.046, pad=0.04)
ax.set_title(f"DLA cluster ($\\eta={ETA}$, {n_cluster} sites)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_aspect("equal")

# 2. Final concentration field
ax = axes[1]
im1 = ax.pcolormesh(X, Y, concentration, shading="nearest",
                    cmap="gist_heat", vmin=0, vmax=1)
fig.colorbar(im1, ax=ax, label="$c(x,y)$", fraction=0.046, pad=0.04)
ax.set_title("Final concentration field")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_aspect("equal")

# 3. BVP iteration count per growth step
ax = axes[2]
ax.plot(n_iters, lw=0.8)
ax.set_xlabel("Growth step")
ax.set_ylabel("SOR iterations")
ax.set_title("SOR iterations per growth step\n(warm-start benefit)")
ax.grid(True, alpha=0.3)

fig.suptitle(f"PDE-based DLA on a {N}×{N} grid")
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / "a2_1_dla.png"
plt.savefig(fname, dpi=150)
print(f"Saved → {fname}")
plt.show()
