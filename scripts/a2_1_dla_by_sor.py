import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.dla import grow_dla_sor


def fixed_bc(k, y):
    """Enforce diffusion BCs after each iteration."""
    apply_diffusion_bc(y)
    return y


# Parameters
N = 50
grid = Grid2D(N=N, L=1.0)
omega = 1.89
eta = 8.0
n_iter = 50
seed = (23, 2)
tol = 1e-3

print(f"DLA: N={N}, n_steps={n_iter}, η={eta}, ω={omega:.4f}")

# Initial guess: zero everywhere, then apply BCs
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

# Track growth order via callback
growth_order = np.full((N + 1, N + 1), np.nan)
growth_order[seed] = 0


def track_growth(step, y, growth_mask):
    """Record growth order for each newly added site."""
    new_sites = growth_mask & np.isnan(growth_order)
    growth_order[new_sites] = step
    if step % 50 == 0:
        print(f"  step {step}/{n_iter}  cluster size={growth_mask.sum()}")


# Run DLA by SOR
result = grow_dla_sor(
    n_iter, seed, eta, c0, omega, tol, post_step=fixed_bc, post_growth=track_growth
)

# Save directory
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# 1. Cluster coloured by growth order (left)
ax_cluster = axes[0]
cluster_display = np.where(np.isnan(growth_order), np.nan, growth_order)
im_cluster = ax_cluster.pcolormesh(
    grid.X,
    grid.Y,
    cluster_display,
    shading="nearest",
    cmap="plasma",
    vmin=0,
    vmax=n_iter,
)
fig.colorbar(im_cluster, ax=ax_cluster, label="Growth step", fraction=0.046, pad=0.04)
ax_cluster.set_xlabel(r"$x$ [m]")
ax_cluster.set_ylabel(r"$y$ [m]")
ax_cluster.set_aspect("equal")
n_sites = np.count_nonzero(~np.isnan(growth_order))
print(f"Done. Cluster size: {n_sites} sites")
ax_cluster.set_title(f"DLA cluster ($\\eta={eta}$) — {n_sites} sites")

# 2. Concentration field (right)
ax_conc = axes[1]
im_conc = ax_conc.pcolormesh(
    grid.X, grid.Y, result.y, shading="nearest", cmap="gist_heat", vmin=0, vmax=1
)
fig.colorbar(im_conc, ax=ax_conc, label=r"$c(x,y)$", fraction=0.046, pad=0.04)
ax_conc.set_xlabel(r"$x$ [m]")
ax_conc.set_ylabel(r"$y$ [m]")
ax_conc.set_aspect("equal")
ax_conc.set_title("Concentration field $c(x, y)$")

fig.suptitle(f"PDE-based DLA on a {N}$\\times${N} grid", fontsize=14)
plt.tight_layout()

filename = "a2_1_dla_by_sor.png"
plt.savefig(out_dir / filename, dpi=150)
print(f"Saved to {out_dir / filename}")

plt.show()
