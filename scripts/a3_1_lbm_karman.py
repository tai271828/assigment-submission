"""Kármán vortex street simulation using the Lattice Boltzmann Method.

Simulates flow past a cylinder at various Reynolds numbers using the
D2Q9 LBM with BGK collision operator.

The setup follows the Schäfer-Turek benchmark:
  - Channel: 2.2m x 0.41m
  - Cylinder center: (0.2m, 0.2m), diameter D = 0.1m
  - Parabolic inlet velocity profile
  - No-slip top/bottom walls
  - Zero-gradient outflow

Usage:
    python scripts/a3_1_lbm_karman.py [--re RE] [--resolution RES] [--steps N]
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from scicomp3.lbm import LBMSolver, cylinder_mask
from scicomp3.lbm.solver import tau_from_re, compute_re


# -- Physical setup (Schäfer-Turek benchmark) --
L_PHYS = 2.2     # channel length [m]
H_PHYS = 0.41    # channel height [m]
CX_PHYS = 0.2    # cylinder center x [m]
CY_PHYS = 0.2    # cylinder center y [m]
D_PHYS = 0.1     # cylinder diameter [m]


def setup_simulation(Re, resolution=20, u_lb=0.04):
    """Create an LBM solver for the Kármán vortex street.

    Parameters:
        Re: Reynolds number based on cylinder diameter and peak inlet velocity.
        resolution: Lattice nodes per cylinder diameter.
        u_lb: Peak inlet velocity in lattice units (keep << cs = 1/sqrt(3) ≈ 0.577).

    Returns:
        solver: Configured LBMSolver instance.
        info: Dict with simulation parameters.
    """
    D_lb = resolution  # cylinder diameter in lattice nodes
    dx = D_PHYS / D_lb  # physical size per lattice node

    Nx = int(L_PHYS / dx) + 1
    Ny = int(H_PHYS / dx) + 1

    cx_lb = int(CX_PHYS / dx)
    cy_lb = int(CY_PHYS / dx)
    r_lb = D_lb / 2.0

    tau = tau_from_re(Re, u_lb, D_lb)
    nu_lb = (tau - 0.5) / 3.0

    obstacle = cylinder_mask(Nx, Ny, cx_lb, cy_lb, r_lb)

    solver = LBMSolver(Nx, Ny, tau, obstacle, u_inlet=u_lb)

    info = {
        "Re": Re,
        "Nx": Nx,
        "Ny": Ny,
        "D_lb": D_lb,
        "cx_lb": cx_lb,
        "cy_lb": cy_lb,
        "r_lb": r_lb,
        "tau": tau,
        "nu_lb": nu_lb,
        "u_lb": u_lb,
        "dx": dx,
        "Re_actual": compute_re(u_lb, D_lb, nu_lb),
    }
    return solver, info


def plot_vorticity(ax, solver, info, title=None):
    """Plot the vorticity field (curl of velocity)."""
    ux = solver.ux.T
    uy = solver.uy.T

    # Vorticity = duy/dx - dux/dy
    dvydx = np.gradient(uy, axis=1)
    dvxdy = np.gradient(ux, axis=0)
    vorticity = dvydx - dvxdy

    # Mask obstacle
    obs = solver.obstacle.T
    vorticity[obs] = np.nan

    vmax = np.nanmax(np.abs(vorticity)) * 0.5
    if vmax < 1e-10:
        vmax = 1e-3

    ax.imshow(
        vorticity,
        cmap="RdBu_r",
        origin="lower",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    # Draw cylinder
    circle = plt.Circle(
        (info["cx_lb"], info["cy_lb"]),
        info["r_lb"],
        color="gray",
        zorder=10,
    )
    ax.add_patch(circle)

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")


def plot_velocity_magnitude(ax, solver, info, title=None):
    """Plot the velocity magnitude field."""
    speed = np.sqrt(solver.ux**2 + solver.uy**2).T
    obs = solver.obstacle.T
    speed[obs] = np.nan

    ax.imshow(
        speed,
        cmap="viridis",
        origin="lower",
        aspect="auto",
    )

    circle = plt.Circle(
        (info["cx_lb"], info["cy_lb"]),
        info["r_lb"],
        color="white",
        zorder=10,
    )
    ax.add_patch(circle)

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")


def run_and_plot(Re, resolution, n_steps, save_dir="images"):
    """Run simulation and save plots."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"Setting up LBM simulation: Re={Re}, resolution={resolution}")
    solver, info = setup_simulation(Re, resolution)
    print(f"  Grid: {info['Nx']} x {info['Ny']}")
    print(f"  tau: {info['tau']:.4f}, nu_lb: {info['nu_lb']:.6f}")
    print(f"  Re (actual): {info['Re_actual']:.1f}")

    if info["tau"] <= 0.5:
        print(f"  WARNING: tau={info['tau']:.4f} <= 0.5 -> unstable!")
        return None, info

    if info["tau"] < 0.505:
        print(f"  WARNING: tau={info['tau']:.4f} very close to 0.5 -> may be unstable!")

    # Save snapshots at intervals
    save_times = [
        n_steps // 5,
        2 * n_steps // 5,
        3 * n_steps // 5,
        4 * n_steps // 5,
        n_steps,
    ]

    print(f"  Running {n_steps} steps...")
    step_count = 0
    snapshots = []

    def save_callback(step, s):
        nonlocal step_count
        step_count = step
        if step in save_times:
            snapshots.append((
                step,
                s.rho.copy(),
                s.ux.copy(),
                s.uy.copy(),
            ))
            print(f"    Step {step}/{n_steps} - max|u|={np.max(np.sqrt(s.ux**2 + s.uy**2)):.6f}")

            # Check stability
            if np.any(np.isnan(s.rho)) or np.any(np.isinf(s.rho)):
                print(f"    DIVERGED at step {step}!")
                raise RuntimeError(f"Simulation diverged at step {step}")
            if np.max(np.abs(s.ux)) > 0.5:
                print(f"    WARNING: velocity approaching lattice sound speed!")

    # Use velocity ramp for first 10% of steps to avoid transient instability
    ramp_steps = max(n_steps // 10, 1000)
    print(f"  Ramping inlet velocity over {ramp_steps} steps")

    try:
        result = solver.solve(n_steps, callback=save_callback, ramp_steps=ramp_steps)
    except RuntimeError as e:
        print(f"  Simulation failed: {e}")
        return None, info

    # Plot final vorticity and velocity
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    plot_vorticity(axes[0], solver, info, f"Vorticity - Re={Re}, step={n_steps}")
    plot_velocity_magnitude(axes[1], solver, info, f"Velocity magnitude - Re={Re}")
    plt.tight_layout()
    fname = os.path.join(save_dir, f"lbm_karman_Re{Re}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

    # Plot time evolution of vorticity
    if len(snapshots) >= 3:
        n_panels = min(len(snapshots), 5)
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels))
        if n_panels == 1:
            axes = [axes]

        for idx, (step, rho, ux, uy) in enumerate(snapshots[:n_panels]):
            # Temporarily set solver fields to snapshot
            solver.ux = ux
            solver.uy = uy
            plot_vorticity(axes[idx], solver, info, f"Step {step}")

        plt.tight_layout()
        fname_evo = os.path.join(save_dir, f"lbm_karman_Re{Re}_evolution.png")
        fig.savefig(fname_evo, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname_evo}")

        # Restore final state
        solver.ux = result.ux
        solver.uy = result.uy

    return result, info


def main():
    parser = argparse.ArgumentParser(description="LBM Kármán vortex street")
    parser.add_argument("--re", type=float, default=100, help="Reynolds number")
    parser.add_argument("--resolution", type=int, default=20,
                        help="Lattice nodes per cylinder diameter")
    parser.add_argument("--steps", type=int, default=20000,
                        help="Number of time steps")
    parser.add_argument("--save-dir", type=str, default="images",
                        help="Directory for output images")
    args = parser.parse_args()

    run_and_plot(args.re, args.resolution, args.steps, args.save_dir)


if __name__ == "__main__":
    main()
