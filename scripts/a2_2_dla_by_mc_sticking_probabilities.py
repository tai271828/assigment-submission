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


# -- Parameters --------------------------------------------------------------
N = 50
N_STEPS = 100
SEED = 42
STICKING_PROBS = (0.1, 0.5, 1.0)

grid = Grid2D(N=N, L=1.0)
growth_seed = (N // 2, N // 2)

print(f"DLA comparison: N={N}, n_steps={N_STEPS}")

# -- Build figure --------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))


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
    fig.colorbar(im, ax=ax, label="Growth step")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_title(f"$p_s = {p}$")

fig.suptitle(
    f"MC-based DLA on a {N}$\\times${N} grid for various values of $p_s$", fontsize=14
)
plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig_path = out_dir / "a2_2_dla_by_mc_sticking_probs.png"
plt.savefig(fig_path, dpi=150)
print(f"Saved → {fig_path}")

plt.show()
