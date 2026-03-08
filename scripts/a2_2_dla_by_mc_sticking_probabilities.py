"""
Comparison of cluster shapes for various sticking probabilities in the
DLA simulation using Monte Carlo random walkers
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
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
plt.rcParams.update({"font.size": 14})


# -- Parameters --------------------------------------------------------------
N = 100
N_STEPS = 300
SEED = 42
STICKING_PROBS = (0.1, 0.5, 1.0)

grid = Grid2D(N=N, L=1.0)
growth_seed = (N // 2, N // 2)

print(f"DLA comparison: N={N}, n_steps={N_STEPS}")

# -- Build figure --------------------------------------------------------
n_rows = len(STICKING_PROBS)
fig, axes = plt.subplots(n_rows, 1, figsize=(4, 4 * n_rows), constrained_layout=True)


for ax, p in zip(axes, STICKING_PROBS):
    # Reset state for each simulation
    growth_order = np.full((N + 1, N + 1), np.nan)
    growth_order[growth_seed] = 0

    def capture_growth(step, growth_mask, candidates):
        """Record growth order after each growth step."""
        new_sites = growth_mask & np.isnan(growth_order)
        growth_order[new_sites] = step
        if step % 10 == 0:
            print(f"  step {step}/{N_STEPS}  cluster size={growth_mask.sum()}")

    np.random.seed(SEED)
    result = grow_dla_mc(N_STEPS, growth_seed, N, p, post_growth=capture_growth)

    n_cluster = result.growth_mask.sum()
    print(f"Done. p_s={p}, cluster size: {n_cluster} sites")

    # Plot final state
    im = ax.pcolormesh(
        grid.X,
        grid.Y,
        growth_order,
        shading="nearest",
        cmap="plasma",
        vmin=0,
        vmax=N_STEPS,
    )
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$y$", fontsize=16)
    ax.set_aspect("equal")
    n_sites = np.count_nonzero(~np.isnan(growth_order))
    ax.set_title(f"$p_s = {p}$  ({n_sites} sites)", fontsize=16)

cbar = fig.colorbar(im, ax=axes[0], location="top", fraction=0.05, pad=0.12)
cbar.set_label("Growth step", fontsize=14)
cbar.ax.tick_params(labelsize=12)

fig.suptitle(
    f"DLA cluster shape vs $p_s$\n{N_STEPS} steps on {N}$\\times${N} grid",
    fontsize=18,
)

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig_path = out_dir / "a2_2_dla_by_mc_sticking_probs.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved → {fig_path}")

plt.show()
