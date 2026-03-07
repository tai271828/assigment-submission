"""DLA growth animation via MC (Assignment 2.2).

Runs a MC-based DLA simulation step by step and animates:
  - Left:  the growing cluster coloured by growth order.
  - Right: the grid with random walkers

The animation is saved as a GIF and also displayed interactively.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from scicomp3.core.grid import Grid2D
from scicomp3.models.dla_by_mc import grow_dla_mc

import scienceplots  # noqa: F401

styles = (
    ["science"]
    if (shutil.which("latex") and shutil.which("dvipng"))
    else ["science", "no-latex"]
)
plt.style.use(styles)


# -- Parameters --------------------------------------------------------------
N = 50
N_STEPS = 100
SEED = 42
STICKING_PROB = 1.0

grid = Grid2D(N=N, L=1.0)
growth_seed = (N // 2, N // 2)

print(f"DLA animation: N={N}, n_steps={N_STEPS}")

# -- Run DLA simulation -----------------------------------------------------
np.random.seed(SEED)

# Capture frames via post_growth callback
growth_order = np.full((N + 1, N + 1), np.nan)
growth_order[growth_seed] = 0
frames = [growth_order.copy()]


def capture_growth(step, growth_mask, candidates):
    """Record growth order and walkers location after each growth step."""
    # Detect newly added site
    new_sites = growth_mask & np.isnan(growth_order)
    growth_order[new_sites] = step
    frames.append(growth_order.copy())
    if step % 10 == 0:
        print(f"  step {step}/{N_STEPS}  cluster size={growth_mask.sum()}")


result = grow_dla_mc(N_STEPS, growth_seed, N, STICKING_PROB, post_growth=capture_growth)

n_cluster = result.growth_mask.sum()
print(f"Done. Cluster size: {n_cluster} sites, {len(frames)} frames")

# -- Build animation --------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

# Left: cluster coloured by growth order
ax_cluster = ax
cluster_display = np.where(np.isnan(frames[0]), np.nan, frames[0])
im_cluster = ax_cluster.pcolormesh(
    grid.X,
    grid.Y,
    cluster_display,
    shading="nearest",
    cmap="plasma",
    vmin=0,
    vmax=N_STEPS,
)
fig.colorbar(im_cluster, ax=ax_cluster, label="Growth step")
ax_cluster.set_xlabel("$x$")
ax_cluster.set_ylabel("$y$")
ax_cluster.set_aspect("equal")
title_cluster = ax_cluster.set_title(f"DLA cluster ($p_s={STICKING_PROB}$) — step 0")

fig.suptitle(f"MC-based DLA on a {N}$\\times${N} grid", fontsize=14)
plt.tight_layout()


def update(frame_idx):
    g_order = frames[frame_idx]
    n_sites = np.count_nonzero(~np.isnan(g_order))

    cluster_display = np.where(np.isnan(g_order), np.nan, g_order)
    im_cluster.set_array(cluster_display.ravel())
    title_cluster.set_text(
        f"DLA cluster ($p_s={STICKING_PROB}$) — step {frame_idx}, {n_sites} sites"
    )

    return im_cluster, title_cluster


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
gif_path = out_dir / "a2_2_dla_by_mc.gif"
print(f"Saving animation ({len(frames)} frames)...")
anim.save(gif_path, writer="pillow", fps=15, dpi=100)
print(f"Saved → {gif_path}")

plt.show()
