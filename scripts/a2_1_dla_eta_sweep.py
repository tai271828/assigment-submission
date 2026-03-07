"""DLA eta sweep via SOR (Assignment 2.1).

Runs DLA simulations for several values of eta and shows the resulting
cluster shapes side by side.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.models.dla_by_sor import grow_dla_sor
from scicomp3.bvp.omega import get_optimal_omega

import scienceplots  # noqa: F401 (registers styles on import)

styles = ["science"] if shutil.which("latex") else ["science", "no-latex"]
plt.style.use(styles)
plt.rcParams.update({"font.size": 14})


def fixed_bc(k, y):
    """Enforce diffusion BCs after each iteration."""
    apply_diffusion_bc(y)
    return y


# -- Parameters --------------------------------------------------------------
N = 100
N_STEPS = 100
SEED = 42
OMEGA = get_optimal_omega(N)
TOL = 1e-4
MAX_ITER = 2_000
ETAS = [0.0, 0.5, 1.0, 2.0]

grid = Grid2D(N=N, L=1.0)
growth_seed = (N // 2, N // 2)

# -- Run DLA for each eta ----------------------------------------------------
results = []
for eta in ETAS:
    np.random.seed(SEED)

    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)

    growth_order = np.full((N + 1, N + 1), np.nan)
    growth_order[growth_seed] = 0

    def make_tracker(go):
        def track_growth(step, y, growth_mask):
            new_sites = growth_mask & np.isnan(go)
            go[new_sites] = step

        return track_growth

    result = grow_dla_sor(
        N_STEPS,
        growth_seed,
        eta,
        c0,
        OMEGA,
        TOL,
        max_iter_sor=MAX_ITER,
        post_step=fixed_bc,
        post_growth=make_tracker(growth_order),
        method="sor_numba",
    )

    n_sites = np.count_nonzero(~np.isnan(growth_order))
    print(f"η={eta:.1f}  cluster size={n_sites}")
    results.append((eta, growth_order.copy(), result.y))

# -- Plotting -----------------------------------------------------------------
n_rows = len(ETAS)
fig, axes = plt.subplots(n_rows, 1, figsize=(4, 4 * n_rows), constrained_layout=True)

for ax, (eta, growth_order, _) in zip(axes, results):
    cluster_display = np.where(np.isnan(growth_order), np.nan, growth_order)
    im = ax.pcolormesh(
        grid.X,
        grid.Y,
        cluster_display,
        shading="nearest",
        cmap="plasma",
        vmin=0,
        vmax=N_STEPS,
    )
    n_sites = np.count_nonzero(~np.isnan(growth_order))
    ax.set_title(f"$\\eta={eta}$  ({n_sites} sites)", fontsize=16)
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$y$", fontsize=16)
    ax.set_aspect("equal")

cbar = fig.colorbar(im, ax=axes[0], location="top", fraction=0.05, pad=0.12)
cbar.set_label("Growth step", fontsize=14)
cbar.ax.tick_params(labelsize=12)
fig.suptitle(
    f"DLA cluster shape vs $\\eta$\n{N_STEPS} steps on {N}$\\times${N} grid",
    fontsize=18,
)
# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
filename = "a2_1_dla_eta_sweep.png"
plt.savefig(out_dir / filename, dpi=150)
print(f"Saved to {out_dir / filename}")

plt.show()
