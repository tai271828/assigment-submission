"""Kármán vortex street via Finite Difference (Assignment 3.1).

Runs the projection-method FD solver at several Reynolds numbers
and plots the vorticity field. Reports maximum stable Re.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from scicomp3.pde.navier_stokes_fd import simulate_ns_fd

# -- Geometry (benchmark channel from Schäfer-Turek) -------------------------
Lx, Ly = 2.2, 0.41
cx, cy = 0.2, 0.2
r_cyl = 0.05
D_cyl = 2 * r_cyl
U_in = 1.0

# -- Grid --------------------------------------------------------------------
Nx, Ny = 220, 41

# -- Reynolds number sweep ---------------------------------------------------
RE_VALUES = [20, 50, 100, 200]
T_SIM = 8.0
SAVE_EVERY = 200

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

results = {}
for Re in RE_VALUES:
    print(f"\n{'='*60}")
    print(f"Re = {Re}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    res = simulate_ns_fd(
        Nx, Ny, Lx, Ly, Re, U_in, D_cyl, cx, cy, r_cyl,
        T=T_SIM, save_every=SAVE_EVERY, verbose=True,
    )
    wall_time = time.perf_counter() - t0
    results[Re] = res
    print(f"  Wall time: {wall_time:.1f}s, stable={res.stable}")

# -- Plot final vorticity for each Re ----------------------------------------
stable_re = [Re for Re, res in results.items() if res.stable]
fig, axes = plt.subplots(len(stable_re), 1, figsize=(12, 3 * len(stable_re)),
                         squeeze=False)

dx = Lx / Nx
dy = Ly / Ny

for ax_row, Re in zip(axes, stable_re):
    ax = ax_row[0]
    res = results[Re]
    u_final = res.u[-1]
    v_final = res.v[-1]

    # Compute vorticity: omega = dv/dx - du/dy
    dvdx = np.gradient(v_final, dx, axis=1)
    dudy = np.gradient(u_final, dy, axis=0)
    vorticity = dvdx - dudy

    x = np.linspace(0, Lx, u_final.shape[1])
    y = np.linspace(0, Ly, u_final.shape[0])
    X, Y = np.meshgrid(x, y)

    vmax = np.percentile(np.abs(vorticity), 98)
    im = ax.pcolormesh(X, Y, vorticity, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="auto")
    # Draw cylinder
    theta = np.linspace(0, 2 * np.pi, 50)
    ax.plot(cx + r_cyl * np.cos(theta), cy + r_cyl * np.sin(theta), 'k-', lw=1.5)
    ax.fill(cx + r_cyl * np.cos(theta), cy + r_cyl * np.sin(theta), 'gray')
    ax.set_aspect("equal")
    ax.set_title(f"Re = {Re}, t = {res.t[-1]:.2f}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, label="Vorticity", shrink=0.8)

fig.suptitle("Kármán Vortex Street — Finite Difference (Projection Method)", fontsize=14)
plt.tight_layout()
plt.savefig(out_dir / "a3_1_ns_fd_vorticity.png", dpi=150)
print(f"\nSaved to {out_dir / 'a3_1_ns_fd_vorticity.png'}")

# -- Summary ------------------------------------------------------------------
print("\n--- Summary ---")
for Re, res in results.items():
    print(f"  Re={Re:>5d}  stable={res.stable}  snapshots={len(res.t)}")
max_stable = max((Re for Re, r in results.items() if r.stable), default=0)
print(f"  Max stable Re (FD): {max_stable}")

plt.show()
