"""Monte Carlo DLA — comparison with PDE DLA (Assignment 2.2.C).

Runs both the PDE (Laplace-equation) DLA and the Monte Carlo random-walker
DLA on the same N×N grid and plots the resulting clusters side-by-side.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots  # noqa: F401

styles = ["science"] if shutil.which("latex") else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.models.dla    import run_dla
from scicomp3.models.mc_dla import run_mc_dla
from scicomp3.bvp.omega     import get_optimal_omega

# ── Parameters ──────────────────────────────────────────────────────────────
N       = 100
N_STEPS = 250
SEED    = 42
OMEGA   = get_optimal_omega(N)

x_grid = np.linspace(0, 1, N + 1)
y_grid = np.linspace(0, 1, N + 1)
X, Y   = np.meshgrid(x_grid, y_grid, indexing="ij")

# ── PDE DLA ──────────────────────────────────────────────────────────────────
print(f"Running PDE DLA  (N={N}, n_steps={N_STEPS}) …", end=" ", flush=True)
rng = np.random.default_rng(SEED)
pde_mask, pde_conc, pde_n_iters = run_dla(
    N=N, n_steps=N_STEPS, eta=1.0, omega=OMEGA, rng=rng
)
print(f"done  ({pde_mask.sum()} sites)")

# ── Monte Carlo DLA ───────────────────────────────────────────────────────────
print(f"Running MC  DLA  (N={N}, n_steps={N_STEPS}) …", end=" ", flush=True)
rng = np.random.default_rng(SEED)
mc_mask, mc_walk_lengths = run_mc_dla(N=N, n_steps=N_STEPS, ps=1.0, rng=rng)
print(f"done  ({mc_mask.sum()} sites), "
      f"mean walk length = {np.mean(mc_walk_lengths):.0f} steps")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, mask, title in [
    (axes[0], pde_mask,  f"PDE DLA ($\\eta=1$)\n{pde_mask.sum()} sites"),
    (axes[1], mc_mask,   f"MC DLA ($p_s=1$)\n{mc_mask.sum()} sites"),
]:
    display = np.where(mask, 1.0, np.nan)
    ax.pcolormesh(X, Y, display, shading="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

# Walk-length distribution
ax = axes[2]
ax.hist(mc_walk_lengths, bins=40, color="steelblue", edgecolor="white", lw=0.4)
ax.set_xlabel("Walk length (steps)")
ax.set_ylabel("Count")
ax.set_title("MC DLA: walk-length distribution")
ax.set_xscale("log")
ax.grid(True, alpha=0.3)

fig.suptitle(f"PDE-DLA vs. MC-DLA on a {N}×{N} grid ({N_STEPS} particles)")
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / "a2_2_mc_dla.png"
plt.savefig(fname, dpi=150)
print(f"Saved → {fname}")
plt.show()
