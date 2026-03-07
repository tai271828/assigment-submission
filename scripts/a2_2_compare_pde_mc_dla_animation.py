"""PDE-DLA vs MC-DLA side-by-side comparison animation (Assignment 2.2).

Grows both clusters step-by-step and animates three subplots:
  - Left:   PDE concentration field
  - Centre: SOR-based DLA cluster (coloured by growth order)
  - Right:  MC-based DLA cluster  (coloured by growth order)

Uses the numba-accelerated SOR solver for the PDE-based DLA.
Saved as a GIF and also displayed interactively.

Usage:
  python a2_2_compare_pde_mc_dla_animation.py [--N 100] [--eta 1.0] [--ps 1.0] [--steps 200] [--seed 42]
"""

import argparse
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

import scienceplots  # noqa: F401

styles = (
    ["science"]
    if (shutil.which("latex") and shutil.which("dvipng"))
    else ["science", "no-latex"]
)
plt.style.use(styles)

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.models.dla_by_sor import grow_dla_sor
from scicomp3.models.dla_by_mc import grow_dla_mc
from scicomp3.bvp.omega import get_optimal_omega


def fixed_bc(k, y):
    """Enforce diffusion BCs after each SOR iteration."""
    apply_diffusion_bc(y)
    return y


# -- Parameters --------------------------------------------------------------
parser = argparse.ArgumentParser(description="PDE-DLA vs MC-DLA comparison animation")
parser.add_argument("--N", type=int, default=50, help="Grid size (default: 50)")
parser.add_argument("--eta", type=float, default=1.0, help="PDE-DLA growth bias (default: 1.0)")
parser.add_argument("--ps", type=float, default=1.0, help="MC-DLA sticking probability (default: 1.0)")
parser.add_argument("--steps", type=int, default=200, help="Number of growth steps (default: 200)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
args = parser.parse_args()

N = args.N
N_STEPS = args.steps
ETA = args.eta
STICKING_PROB = args.ps
SEED = args.seed
OMEGA = get_optimal_omega(N)
TOL = 1e-4
MAX_ITER = 2_000
BVP_METHOD = "sor_numba"

grid = Grid2D(N=N, L=1.0)
growth_seed = (N // 2, N // 2)

print(f"Compare DLA animation: N={N}, n_steps={N_STEPS}, eta={ETA}, ps={STICKING_PROB}")
print(f"BVP method: {BVP_METHOD}, omega={OMEGA:.4f}")

# -- Run SOR-based DLA simulation -------------------------------------------
print("\nRunning SOR-based DLA...")
np.random.seed(SEED)

c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

sor_growth_order = np.full((N + 1, N + 1), np.nan)
sor_growth_order[growth_seed] = 0
sor_frames = [(sor_growth_order.copy(), c0.copy())]


def capture_sor_growth(step, y, growth_mask):
    """Record growth order and concentration after each SOR growth step."""
    new_sites = growth_mask & np.isnan(sor_growth_order)
    sor_growth_order[new_sites] = step
    sor_frames.append((sor_growth_order.copy(), y.copy()))
    if step % 50 == 0:
        print(f"  SOR step {step}/{N_STEPS}  cluster size={growth_mask.sum()}")


t0 = time.perf_counter()
sor_result = grow_dla_sor(
    N_STEPS,
    growth_seed,
    ETA,
    c0,
    OMEGA,
    TOL,
    max_iter_sor=MAX_ITER,
    post_step=fixed_bc,
    post_growth=capture_sor_growth,
    method=BVP_METHOD,
)
sor_elapsed = time.perf_counter() - t0
print(f"SOR done. Cluster size: {sor_result.growth_mask.sum()} sites, "
      f"{len(sor_frames)} frames, {sor_elapsed:.1f}s")

# -- Run MC-based DLA simulation --------------------------------------------
print("\nRunning MC-based DLA...")
np.random.seed(SEED)

mc_growth_order = np.full((N + 1, N + 1), np.nan)
mc_growth_order[growth_seed] = 0
mc_frames = [mc_growth_order.copy()]


def capture_mc_growth(step, growth_mask, candidates):
    """Record growth order after each MC growth step."""
    new_sites = growth_mask & np.isnan(mc_growth_order)
    mc_growth_order[new_sites] = step
    mc_frames.append(mc_growth_order.copy())
    if step % 50 == 0:
        print(f"  MC step {step}/{N_STEPS}  cluster size={growth_mask.sum()}")


t0 = time.perf_counter()
mc_result = grow_dla_mc(
    N_STEPS,
    growth_seed,
    N,
    STICKING_PROB,
    post_growth=capture_mc_growth,
)
mc_elapsed = time.perf_counter() - t0
print(f"MC done. Cluster size: {mc_result.growth_mask.sum()} sites, "
      f"{len(mc_frames)} frames, {mc_elapsed:.1f}s")

# -- Combine frames ----------------------------------------------------------
# Both simulations produce N_STEPS + 1 frames (initial + one per step).
n_frames = min(len(sor_frames), len(mc_frames))
print(f"\nTotal frames for animation: {n_frames}")

# -- Build animation ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Left: concentration field
ax_conc = axes[0]
im_conc = ax_conc.pcolormesh(
    grid.X, grid.Y, sor_frames[0][1],
    shading="nearest", cmap="gist_heat", vmin=0, vmax=1,
)
fig.colorbar(im_conc, ax=ax_conc, label="$c(x,y)$", fraction=0.046, pad=0.04)
ax_conc.set_xlabel("$x$")
ax_conc.set_ylabel("$y$")
ax_conc.set_aspect("equal")
title_conc = ax_conc.set_title("Concentration field")

# Centre: SOR-based DLA cluster
ax_sor = axes[1]
sor_display = np.where(np.isnan(sor_frames[0][0]), np.nan, sor_frames[0][0])
im_sor = ax_sor.pcolormesh(
    grid.X, grid.Y, sor_display,
    shading="nearest", cmap="plasma", vmin=0, vmax=N_STEPS,
)
fig.colorbar(im_sor, ax=ax_sor, label="Growth step", fraction=0.046, pad=0.04)
ax_sor.set_xlabel("$x$")
ax_sor.set_ylabel("$y$")
ax_sor.set_aspect("equal")
title_sor = ax_sor.set_title(f"SOR DLA ($\\eta={ETA}$) — step 0")

# Right: MC-based DLA cluster
ax_mc = axes[2]
mc_display = np.where(np.isnan(mc_frames[0]), np.nan, mc_frames[0])
im_mc = ax_mc.pcolormesh(
    grid.X, grid.Y, mc_display,
    shading="nearest", cmap="viridis", vmin=0, vmax=N_STEPS,
)
fig.colorbar(im_mc, ax=ax_mc, label="Growth step", fraction=0.046, pad=0.04)
ax_mc.set_xlabel("$x$")
ax_mc.set_ylabel("$y$")
ax_mc.set_aspect("equal")
title_mc = ax_mc.set_title(f"MC DLA ($p_s={STICKING_PROB}$) — step 0")

fig.suptitle(f"PDE-DLA vs MC-DLA on a {N}$\\times${N} grid", fontsize=14)
plt.tight_layout()


def update(frame_idx):
    sor_order, conc = sor_frames[frame_idx]
    mc_order = mc_frames[frame_idx]

    n_sor = np.count_nonzero(~np.isnan(sor_order))
    n_mc = np.count_nonzero(~np.isnan(mc_order))

    im_conc.set_array(conc.ravel())
    title_conc.set_text(f"Concentration field — step {frame_idx}")

    im_sor.set_array(np.where(np.isnan(sor_order), np.nan, sor_order).ravel())
    title_sor.set_text(f"SOR DLA ($\\eta={ETA}$) — step {frame_idx}, {n_sor} sites")

    im_mc.set_array(np.where(np.isnan(mc_order), np.nan, mc_order).ravel())
    title_mc.set_text(f"MC DLA ($p_s={STICKING_PROB}$) — step {frame_idx}, {n_mc} sites")

    return im_conc, im_sor, im_mc, title_conc, title_sor, title_mc


anim = FuncAnimation(
    fig, update, frames=n_frames, interval=50, blit=False, repeat=True,
)

# Save as GIF
out_dir = Path(__file__).parent.parent / "images" / "gifs"
out_dir.mkdir(parents=True, exist_ok=True)
gif_name = f"a2_2_compare_pde_mc_dla_N{N}_eta{ETA}_ps{STICKING_PROB}.gif"
gif_path = out_dir / gif_name
print(f"Saving animation ({n_frames} frames)...")
anim.save(gif_path, writer="pillow", fps=15, dpi=100)
print(f"Saved → {gif_path}")

plt.show()
