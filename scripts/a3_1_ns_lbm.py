"""Kármán vortex street via Lattice Boltzmann Method (Assignment 3.1).

Runs the D2Q9 BGK LBM solver at several Reynolds numbers
and plots the vorticity field. Reports maximum stable Re.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from scicomp3.pde.lattice_boltzmann import simulate_lbm

# -- Geometry in lattice units -----------------------------------------------
# Map physical domain to lattice: Lx=2.2m, Ly=0.41m, D=0.1m
# Choose resolution so D_cyl maps to ~20 lattice nodes
D_lattice = 20  # cylinder diameter in lattice units
r_lattice = D_lattice // 2

# Physical to lattice scaling
Lx_phys, Ly_phys = 2.2, 0.41
D_phys = 0.1
scale = D_lattice / D_phys  # lattice units per meter

Nx = int(Lx_phys * scale)
Ny = int(Ly_phys * scale)
cx = int(0.2 * scale)
cy = Ny // 2

U_in = 0.04  # lattice velocity (keep << 1/sqrt(3) ≈ 0.577)

# -- Reynolds number sweep ---------------------------------------------------
RE_VALUES = [20, 50, 100, 200, 500]
N_STEPS = 50000
SAVE_EVERY = 1000

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

results = {}
for Re in RE_VALUES:
    print(f"\n{'='*60}")
    print(f"Re = {Re}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    res = simulate_lbm(
        Nx, Ny, Re, U_in, cx, cy, r_lattice, D_lattice,
        n_steps=N_STEPS, save_every=SAVE_EVERY, verbose=True,
    )
    wall_time = time.perf_counter() - t0
    results[Re] = res
    print(f"  Wall time: {wall_time:.1f}s, stable={res.stable}")

# -- Plot final vorticity for each Re ----------------------------------------
stable_re = [Re for Re, res in results.items() if res.stable]
fig, axes = plt.subplots(len(stable_re), 1, figsize=(12, 2.5 * len(stable_re)),
                         squeeze=False)

for ax_row, Re in zip(axes, stable_re):
    ax = ax_row[0]
    res = results[Re]
    u_final = res.u[-1]
    v_final = res.v[-1]

    # Vorticity
    dvdx = np.gradient(v_final, axis=1)
    dudy = np.gradient(u_final, axis=0)
    vorticity = dvdx - dudy

    vmax = np.percentile(np.abs(vorticity), 98)
    if vmax == 0:
        vmax = 1e-6
    im = ax.imshow(vorticity, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="lower", aspect="auto")

    # Draw cylinder
    theta = np.linspace(0, 2 * np.pi, 50)
    ax.plot(cx + r_lattice * np.cos(theta), cy + r_lattice * np.sin(theta),
            'k-', lw=1)
    ax.set_title(f"Re = {Re} (LBM, step {int(res.t[-1])})")
    ax.set_xlabel("x [lattice units]")
    ax.set_ylabel("y [lattice units]")
    fig.colorbar(im, ax=ax, label="Vorticity", shrink=0.8)

fig.suptitle("Kármán Vortex Street — Lattice Boltzmann (D2Q9 BGK)", fontsize=14)
plt.tight_layout()
plt.savefig(out_dir / "a3_1_ns_lbm_vorticity.png", dpi=150)
print(f"\nSaved to {out_dir / 'a3_1_ns_lbm_vorticity.png'}")

# -- Summary ------------------------------------------------------------------
print("\n--- Summary ---")
for Re, res in results.items():
    print(f"  Re={Re:>5d}  stable={res.stable}  snapshots={len(res.t)}")
max_stable = max((Re for Re, r in results.items() if r.stable), default=0)
print(f"  Max stable Re (LBM): {max_stable}")

plt.show()
