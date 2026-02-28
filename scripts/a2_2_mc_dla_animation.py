"""PDE-DLA vs MC-DLA side-by-side animation (Assignment 2.2).

Grows both clusters step-by-step and animates:
  - Left:   PDE DLA cluster (coloured by growth order)
  - Centre: MC DLA cluster  (coloured by growth order)
  - Right:  PDE concentration field

Saved as a GIF and also displayed interactively.
"""

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

from scicomp3.bvp.solver import solve_bvp
from scicomp3.bvp.omega import get_optimal_omega
from scicomp3.models.dla import (
    find_growth_candidates,
    compute_growth_probabilities,
    _bvp_bc,
)
from scicomp3.models.mc_dla import _is_adjacent_to_cluster

# ── Parameters ──────────────────────────────────────────────────────────────
N = 50
N_STEPS = 200
ETA = 1.0
PS = 1.0  # MC sticking probability
SEED = 42
OMEGA = get_optimal_omega(N)
TOL = 1e-4
MAX_ITER = 2_000
BVP_METHOD = "sor_redblack"  # fast vectorised solver for animation
SAVE_EVERY = 1

print(f"DLA animation: N={N}, n_steps={N_STEPS}, η={ETA}, ps={PS}, ω={OMEGA:.4f}")
print(f"BVP method: {BVP_METHOD}")

# ── Initialise both models ──────────────────────────────────────────────────
rng_pde = np.random.default_rng(SEED)
rng_mc = np.random.default_rng(SEED)

# PDE DLA state
pde_mask = np.zeros((N + 1, N + 1), dtype=bool)
pde_mask[N // 2, 1] = True
y_coords = np.linspace(0.0, 1.0, N + 1)
concentration = np.broadcast_to(y_coords, (N + 1, N + 1)).copy()
pde_order = np.full((N + 1, N + 1), np.nan)
pde_order[N // 2, 1] = 0

# MC DLA state
mc_mask = np.zeros((N + 1, N + 1), dtype=bool)
mc_mask[N // 2, 1] = True
mc_order = np.full((N + 1, N + 1), np.nan)
mc_order[N // 2, 1] = 0

# MC direction vectors
dx_mc = np.array([0, 0, -1, 1])
dy_mc = np.array([1, -1, 0, 0])
max_steps_per_walker = 10 * (N + 1) ** 2

# Frames: (pde_order, mc_order, concentration)
frames = [(pde_order.copy(), mc_order.copy(), concentration.copy())]

# ── Grow both clusters step-by-step ─────────────────────────────────────────
print("Running dual DLA simulation...")
t0 = time.perf_counter()

for step in range(1, N_STEPS + 1):
    # --- PDE DLA step ---
    sink_coords = np.argwhere(pde_mask)
    result = solve_bvp(
        concentration,
        method=BVP_METHOD,
        tol=TOL,
        max_iter=MAX_ITER,
        post_step=_bvp_bc,
        sink_coordinates=sink_coords,
        omega=OMEGA,
    )
    concentration = result.y

    candidates = find_growth_candidates(pde_mask)
    if np.any(candidates):
        candidate_indices = np.argwhere(candidates)
        pg = compute_growth_probabilities(concentration, candidates, ETA)
        choice_idx = rng_pde.choice(len(candidate_indices), p=pg)
        new_point = tuple(candidate_indices[choice_idx])
        pde_mask[new_point] = True
        pde_order[new_point] = step

    # --- MC DLA step (add one particle) ---
    mc_added = False
    while not mc_added:
        wx = int(rng_mc.integers(0, N + 1))
        wy = N
        step_count = 0

        while step_count < max_steps_per_walker:
            d = int(rng_mc.integers(0, 4))
            new_wx = (wx + dx_mc[d]) % (N + 1)
            new_wy = wy + dy_mc[d]

            if new_wy > N or new_wy < 0:
                break  # walker escaped

            if mc_mask[new_wx, new_wy]:
                step_count += 1
                continue

            wx, wy = new_wx, new_wy
            step_count += 1

            if _is_adjacent_to_cluster(wx, wy, mc_mask, N):
                if PS >= 1.0 or rng_mc.random() < PS:
                    mc_mask[wx, wy] = True
                    mc_order[wx, wy] = step
                    mc_added = True
                    break

    # --- Save frame ---
    if step % SAVE_EVERY == 0:
        frames.append((pde_order.copy(), mc_order.copy(), concentration.copy()))

    if step % 50 == 0:
        elapsed = time.perf_counter() - t0
        print(
            f"  step {step}/{N_STEPS}  "
            f"PDE:{pde_mask.sum()} sites  MC:{mc_mask.sum()} sites  "
            f"[{elapsed:.1f}s]"
        )

elapsed = time.perf_counter() - t0
print(
    f"Done. PDE:{pde_mask.sum()} sites, MC:{mc_mask.sum()} sites, "
    f"{len(frames)} frames, {elapsed:.1f}s"
)

# ── Build animation ──────────────────────────────────────────────────────────
x_grid = np.linspace(0, 1, N + 1)
y_grid = np.linspace(0, 1, N + 1)
X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Left: PDE DLA cluster
ax_pde = axes[0]
pde_display = np.where(np.isnan(frames[0][0]), np.nan, frames[0][0])
im_pde = ax_pde.pcolormesh(
    X, Y, pde_display, shading="nearest", cmap="plasma", vmin=0, vmax=N_STEPS
)
fig.colorbar(im_pde, ax=ax_pde, label="Growth step", fraction=0.046, pad=0.04)
ax_pde.set_xlabel("$x$")
ax_pde.set_ylabel("$y$")
ax_pde.set_aspect("equal")
title_pde = ax_pde.set_title(f"PDE DLA ($\\eta={ETA}$) — step 0")

# Centre: MC DLA cluster
ax_mc = axes[1]
mc_display = np.where(np.isnan(frames[0][1]), np.nan, frames[0][1])
im_mc = ax_mc.pcolormesh(
    X, Y, mc_display, shading="nearest", cmap="viridis", vmin=0, vmax=N_STEPS
)
fig.colorbar(im_mc, ax=ax_mc, label="Growth step", fraction=0.046, pad=0.04)
ax_mc.set_xlabel("$x$")
ax_mc.set_ylabel("$y$")
ax_mc.set_aspect("equal")
title_mc = ax_mc.set_title(f"MC DLA ($p_s={PS}$) — step 0")

# Right: PDE concentration field
ax_conc = axes[2]
im_conc = ax_conc.pcolormesh(
    X, Y, frames[0][2], shading="nearest", cmap="gist_heat", vmin=0, vmax=1
)
fig.colorbar(im_conc, ax=ax_conc, label="$c(x,y)$", fraction=0.046, pad=0.04)
ax_conc.set_xlabel("$x$")
ax_conc.set_ylabel("$y$")
ax_conc.set_aspect("equal")
title_conc = ax_conc.set_title("PDE concentration")

fig.suptitle(f"PDE-DLA vs MC-DLA on a {N}×{N} grid", fontsize=14)
plt.tight_layout()


def update(frame_idx):
    pde_g, mc_g, conc = frames[frame_idx]
    step_num = frame_idx * SAVE_EVERY
    n_pde = np.count_nonzero(~np.isnan(pde_g))
    n_mc = np.count_nonzero(~np.isnan(mc_g))

    im_pde.set_array(np.where(np.isnan(pde_g), np.nan, pde_g).ravel())
    title_pde.set_text(f"PDE DLA ($\\eta={ETA}$) — step {step_num}, {n_pde} sites")

    im_mc.set_array(np.where(np.isnan(mc_g), np.nan, mc_g).ravel())
    title_mc.set_text(f"MC DLA ($p_s={PS}$) — step {step_num}, {n_mc} sites")

    im_conc.set_array(conc.ravel())
    title_conc.set_text(f"PDE concentration — step {step_num}")

    return im_pde, im_mc, im_conc, title_pde, title_mc, title_conc


anim = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=50,
    blit=False,
    repeat=True,
)

# Save
out_dir = Path(__file__).parent.parent / "images" / "gifs"
out_dir.mkdir(parents=True, exist_ok=True)
gif_path = out_dir / "a2_2_mc_dla_animation.gif"
print(f"Saving animation ({len(frames)} frames)...")
anim.save(gif_path, writer="pillow", fps=15, dpi=100)
print(f"Saved → {gif_path}")

plt.show()
