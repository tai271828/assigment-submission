"""DLA growth animation via SOR (Assignment 2.1).

Runs a PDE-based DLA simulation step by step and animates:
  - Left:  the growing cluster coloured by growth order.
  - Right: the concentration field with cluster overlay.

The animation is saved as a GIF and also displayed interactively.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.models.dla_by_sor import grow_dla_sor
from scicomp3.bvp.omega import get_optimal_omega

import scienceplots  # noqa: F401

styles = (
    ["science"]
    if (shutil.which("latex") and shutil.which("dvipng"))
    else ["science", "no-latex"]
)
plt.style.use(styles)


def fixed_bc(k, y):
    """Enforce diffusion BCs after each iteration."""
    apply_diffusion_bc(y)
    return y


# -- Parameters --------------------------------------------------------------
N = 50
N_STEPS = 100
ETA = 1.0
SEED = 42
OMEGA = get_optimal_omega(N)
TOL = 1e-4
MAX_ITER = 2_000

grid = Grid2D(N=N, L=1.0)
growth_seed = (N // 2, N // 2)

print(f"DLA animation: N={N}, n_steps={N_STEPS}, η={ETA}, ω={OMEGA:.4f}")

# -- Run DLA simulation -----------------------------------------------------
np.random.seed(SEED)

c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

# Capture frames via post_growth callback
growth_order = np.full((N + 1, N + 1), np.nan)
growth_order[growth_seed] = 0
frames = [(growth_order.copy(), c0.copy())]


def capture_growth(step, y, growth_mask):
    """Record growth order and concentration after each growth step."""
    # Detect newly added site
    new_sites = growth_mask & np.isnan(growth_order)
    growth_order[new_sites] = step
    frames.append((growth_order.copy(), y.copy()))
    if step % 10 == 0:
        print(f"  step {step}/{N_STEPS}  cluster size={growth_mask.sum()}")


result = grow_dla_sor(
    N_STEPS,
    growth_seed,
    ETA,
    c0,
    OMEGA,
    TOL,
    max_iter_sor=MAX_ITER,
    post_step=fixed_bc,
    post_growth=capture_growth,
)

n_cluster = result.growth_mask.sum()
print(f"Done. Cluster size: {n_cluster} sites, {len(frames)} frames")

# -- Build animation --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: cluster coloured by growth order
ax_cluster = axes[0]
cluster_display = np.where(np.isnan(frames[0][0]), np.nan, frames[0][0])
im_cluster = ax_cluster.pcolormesh(
    grid.X,
    grid.Y,
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
    grid.X,
    grid.Y,
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

fig.suptitle(f"PDE-based DLA on a {N}$\\times${N} grid", fontsize=14)
plt.tight_layout()


def update(frame_idx):
    g_order, conc = frames[frame_idx]
    n_sites = np.count_nonzero(~np.isnan(g_order))

    cluster_display = np.where(np.isnan(g_order), np.nan, g_order)
    im_cluster.set_array(cluster_display.ravel())
    title_cluster.set_text(
        f"DLA cluster ($\\eta={ETA}$) — step {frame_idx}, {n_sites} sites"
    )

    im_conc.set_array(conc.ravel())
    title_conc.set_text(f"Concentration field — step {frame_idx}")

    return im_cluster, im_conc, title_cluster, title_conc


anim = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=50,
    blit=False,
    repeat=True,
)

# Save as GIF
out_dir = Path(__file__).parent.parent / "images" / "gifs"
out_dir.mkdir(parents=True, exist_ok=True)
gif_path = out_dir / "a2_1_dla_by_sor.gif"
print(f"Saving animation ({len(frames)} frames)...")
anim.save(gif_path, writer="pillow", fps=15, dpi=100)
print(f"Saved → {gif_path}")

plt.show()
