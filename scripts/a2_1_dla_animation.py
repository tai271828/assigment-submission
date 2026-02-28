"""DLA growth animation using the Laplace equation (Assignment 2.1).

Runs a PDE-based DLA simulation step by step and animates:
  - Left:  the growing cluster coloured by growth order.
  - Right: the concentration field with cluster overlay.

The animation is saved as a GIF and also displayed interactively.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from pathlib import Path

import scienceplots  # noqa: F401

styles = (
    ["science"]
    if (shutil.which("latex") and shutil.which("dvipng"))
    else ["science", "no-latex"]
)
plt.style.use(styles)

from scicomp3.bvp.solver import solve_bvp
from scicomp3.bvp.omega import get_optimal_omega
from scicomp3.models.dla import (
    find_growth_candidates,
    compute_growth_probabilities,
    _bvp_bc,
)

# ── Parameters ──────────────────────────────────────────────────────────────
# For quick visual inspection use N=50, n_steps=200 (~1 min).
# For publication quality use N=100, n_steps=500 (~10–15 min).
N = 50
N_STEPS = 200
ETA = 1.0
SEED = 42
OMEGA = get_optimal_omega(N)
TOL = 1e-4
MAX_ITER = 2_000

print(f"DLA animation: N={N}, n_steps={N_STEPS}, η={ETA}, ω={OMEGA:.4f}")

# ── Pre-compute all growth steps ─────────────────────────────────────────────
# Store cluster snapshots at selected frames for the animation.
rng = np.random.default_rng(SEED)

cluster_mask = np.zeros((N + 1, N + 1), dtype=bool)
cluster_mask[N // 2, 1] = True  # seed

y_coords = np.linspace(0.0, 1.0, N + 1)
concentration = np.broadcast_to(y_coords, (N + 1, N + 1)).copy()

# growth_order[i,j] = step number when site (i,j) joined the cluster (0 = seed)
growth_order = np.full((N + 1, N + 1), np.nan)
growth_order[N // 2, 1] = 0

# Store frames: (growth_order snapshot, concentration snapshot)
# Save every frame for smooth animation (or subsample for speed)
SAVE_EVERY = 1  # save every N-th step
frames = [(growth_order.copy(), concentration.copy())]

print("Running DLA simulation...")
for step in range(1, N_STEPS + 1):
    sink_coords = np.argwhere(cluster_mask)

    result = solve_bvp(
        concentration,
        method="sor",
        tol=TOL,
        max_iter=MAX_ITER,
        post_step=_bvp_bc,
        sink_coordinates=sink_coords,
        omega=OMEGA,
    )
    concentration = result.y

    candidates = find_growth_candidates(cluster_mask)
    if not np.any(candidates):
        print(f"Cluster reached boundary at step {step}")
        break

    candidate_indices = np.argwhere(candidates)
    pg = compute_growth_probabilities(concentration, candidates, ETA)
    choice_idx = rng.choice(len(candidate_indices), p=pg)
    new_point = tuple(candidate_indices[choice_idx])
    cluster_mask[new_point] = True
    growth_order[new_point] = step

    if step % SAVE_EVERY == 0:
        frames.append((growth_order.copy(), concentration.copy()))

    if step % 50 == 0:
        print(
            f"  step {step}/{N_STEPS}  cluster size={cluster_mask.sum()}  "
            f"SOR iters={result.n_iter}"
        )

n_cluster = cluster_mask.sum()
print(f"Done. Cluster size: {n_cluster} sites, {len(frames)} frames")

# ── Build animation ──────────────────────────────────────────────────────────
x_grid = np.linspace(0, 1, N + 1)
y_grid = np.linspace(0, 1, N + 1)
X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: cluster coloured by growth order
ax_cluster = axes[0]
cluster_display = np.where(np.isnan(frames[0][0]), np.nan, frames[0][0])
im_cluster = ax_cluster.pcolormesh(
    X,
    Y,
    cluster_display,
    shading="nearest",
    cmap="plasma",
    vmin=0,
    vmax=N_STEPS,
)
fig.colorbar(im_cluster, ax=ax_cluster, label="Growth step", fraction=0.046, pad=0.04)
ax_cluster.set_xlabel("$x$")
ax_cluster.set_ylabel("$y$")
ax_cluster.set_aspect("equal")
title_cluster = ax_cluster.set_title(f"DLA cluster ($\\eta={ETA}$) — step 0")

# Right: concentration field
ax_conc = axes[1]
im_conc = ax_conc.pcolormesh(
    X,
    Y,
    frames[0][1],
    shading="nearest",
    cmap="gist_heat",
    vmin=0,
    vmax=1,
)
fig.colorbar(im_conc, ax=ax_conc, label="$c(x,y)$", fraction=0.046, pad=0.04)
ax_conc.set_xlabel("$x$")
ax_conc.set_ylabel("$y$")
ax_conc.set_aspect("equal")
title_conc = ax_conc.set_title("Concentration field")

fig.suptitle(f"PDE-based DLA on a {N}×{N} grid", fontsize=14)
plt.tight_layout()


def update(frame_idx):
    g_order, conc = frames[frame_idx]
    step_num = frame_idx * SAVE_EVERY
    n_sites = np.count_nonzero(~np.isnan(g_order))

    # Update cluster image
    cluster_display = np.where(np.isnan(g_order), np.nan, g_order)
    im_cluster.set_array(cluster_display.ravel())
    title_cluster.set_text(
        f"DLA cluster ($\\eta={ETA}$) — step {step_num}, {n_sites} sites"
    )

    # Update concentration image
    im_conc.set_array(conc.ravel())
    title_conc.set_text(f"Concentration field — step {step_num}")

    return im_cluster, im_conc, title_cluster, title_conc


anim = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=50,  # ms between frames
    blit=False,
    repeat=True,
)

# Save as GIF
out_dir = Path(__file__).parent.parent / "images" / "gifs"
out_dir.mkdir(parents=True, exist_ok=True)
gif_path = out_dir / "a2_1_dla_animation.gif"
print(f"Saving animation ({len(frames)} frames)...")
anim.save(gif_path, writer="pillow", fps=15, dpi=100)
print(f"Saved → {gif_path}")

plt.show()
