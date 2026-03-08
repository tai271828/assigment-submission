"""Lattice Boltzmann Method (D2Q9 BGK) for 2D incompressible flow.

Simulates the Navier-Stokes equations via a mesoscopic approach:
particle distribution functions f_i evolve through collision and streaming
on a D2Q9 lattice. Macroscopic density and velocity are recovered as
moments of the distributions.

The BGK collision operator relaxes distributions toward equilibrium:
    f_i(x + e_i, t+1) = f_i(x,t) - (1/tau) * (f_i - f_i^eq)

Kinematic viscosity: nu = (tau - 0.5) / 3  (in lattice units)
"""

import numpy as np


# D2Q9 lattice constants
#   0: rest, 1-4: axis-aligned, 5-8: diagonal
E = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ]
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
# Opposite direction indices (for bounce-back)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
CS2 = 1.0 / 3.0  # speed of sound squared


def equilibrium(rho, ux, uy):
    """Compute equilibrium distribution f_eq for D2Q9.

    Args:
        rho: density field (Ny, Nx)
        ux, uy: velocity components (Ny, Nx)

    Returns:
        f_eq: shape (9, Ny, Nx)
    """
    feq = np.zeros((9, *rho.shape))
    usq = ux**2 + uy**2
    for i in range(9):
        eu = E[i, 0] * ux + E[i, 1] * uy
        feq[i] = W[i] * rho * (1.0 + eu / CS2 + eu**2 / (2 * CS2**2) - usq / (2 * CS2))
    return feq


def create_cylinder_mask_lbm(Nx, Ny, cx, cy, r):
    """Create boolean obstacle mask on the LBM grid.

    Args:
        Nx, Ny: lattice dimensions
        cx, cy: cylinder center (in lattice units)
        r: cylinder radius (in lattice units)

    Returns:
        obstacle: boolean array (Ny, Nx), True inside cylinder
    """
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y)
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r**2


def simulate_lbm(
    Nx,
    Ny,
    Re,
    U_in,
    cx,
    cy,
    r_cyl,
    D_cyl,
    n_steps,
    save_every=100,
    verbose=True,
):
    """Run a Lattice Boltzmann simulation of flow past a cylinder.

    All quantities are in lattice units. The physical-to-lattice mapping is:
        - D_cyl lattice nodes = cylinder diameter
        - U_in = inlet velocity in lattice units (keep << 1/sqrt(3) for stability)
        - tau = 3*nu + 0.5, where nu = U_in * D_cyl / Re

    Args:
        Nx, Ny: Lattice size
        Re: Reynolds number
        U_in: Inlet velocity (lattice units, should be << 0.1 for accuracy)
        cx, cy: Cylinder center (lattice units)
        r_cyl: Cylinder radius (lattice units)
        D_cyl: Cylinder diameter (lattice units, for Re calculation)
        n_steps: Number of time steps
        save_every: Save snapshot interval
        verbose: Print progress

    Returns:
        NSResult with velocity and pressure fields
    """
    from ..core.result import NSResult

    nu = U_in * D_cyl / Re
    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau

    if verbose:
        print(f"LBM: Nx={Nx}, Ny={Ny}, Re={Re:.0f}")
        print(f"  U_in={U_in:.4f}, nu={nu:.6f}, tau={tau:.4f}, omega={omega:.4f}")
        print(f"  Ma={U_in / np.sqrt(CS2):.4f}")

    if tau <= 0.5:
        raise ValueError(f"tau={tau:.4f} <= 0.5: unstable. Reduce U_in or increase Re.")

    # Obstacle mask
    obstacle = create_cylinder_mask_lbm(Nx, Ny, cx, cy, r_cyl)

    # Initialize with equilibrium at rest + inlet velocity
    rho = np.ones((Ny, Nx))
    ux = np.full((Ny, Nx), U_in)
    uy = np.zeros((Ny, Nx))
    ux[obstacle] = 0.0
    uy[obstacle] = 0.0
    f = equilibrium(rho, ux, uy)

    # Storage
    saved_t = [0.0]
    saved_u = [ux.copy()]
    saved_v = [uy.copy()]
    saved_p = [(rho * CS2).copy()]

    # Solid mask: obstacle + top/bottom walls
    wall = np.zeros((Ny, Nx), dtype=bool)
    wall[0, :] = True
    wall[-1, :] = True
    solid = obstacle | wall

    stable = True
    for step in range(1, n_steps + 1):
        # Collision (BGK)
        feq = equilibrium(rho, ux, uy)
        f_out = f - omega * (f - feq)

        # Streaming
        f_new = np.zeros_like(f)
        for i in range(9):
            f_new[i] = np.roll(np.roll(f_out[i], E[i, 0], axis=1), E[i, 1], axis=0)

        # Bounce-back on all solid nodes (cylinder + walls)
        for i in range(9):
            f_new[i][solid] = f_out[OPP[i]][solid]

        # Zou-He inlet BC (left wall, x=0): prescribed velocity
        # Only apply on fluid nodes (not on wall corners)
        _zou_he_inlet(f_new, U_in, Ny, wall)

        # Outlet BC (right wall, x=Nx-1): zero-gradient extrapolation
        for i in range(9):
            f_new[i, 1:-1, -1] = f_new[i, 1:-1, -2]

        f = f_new

        # Macroscopic quantities
        rho = f.sum(axis=0)
        rho[rho == 0] = 1.0  # prevent division by zero on solid nodes
        ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho
        uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho
        ux[solid] = 0.0
        uy[solid] = 0.0

        # Stability check
        max_vel = np.sqrt(ux**2 + uy**2).max()
        if np.isnan(max_vel) or max_vel > 0.5:
            if verbose:
                print(f"  Blowup at step {step}, max_vel={max_vel:.4f}")
            stable = False
            break

        if step % save_every == 0 or step == n_steps:
            saved_t.append(float(step))
            saved_u.append(ux.copy())
            saved_v.append(uy.copy())
            saved_p.append((rho * CS2).copy())
            if verbose and step % (save_every * 10) == 0:
                print(f"  step {step}/{n_steps}, max|u|={max_vel:.4f}")

    if verbose:
        print(f"  Done. stable={stable}")

    return NSResult(
        t=np.array(saved_t),
        u=np.array(saved_u),
        v=np.array(saved_v),
        p=np.array(saved_p),
        Re=Re,
        method="lbm",
        stable=stable,
    )


def _zou_he_inlet(f, U_in, Ny, wall):
    """Apply Zou-He velocity BC at the left boundary (x=0).

    Prescribes ux = U_in, uy = 0 at x=0 on fluid nodes only.
    """
    # Slice for fluid rows at inlet (skip wall rows)
    s = slice(1, Ny - 1)  # skip first and last row (walls)
    rho_in = (f[0, s, 0] + f[2, s, 0] + f[4, s, 0]
              + 2 * (f[3, s, 0] + f[6, s, 0] + f[7, s, 0])) / (1 - U_in)
    f[1, s, 0] = f[3, s, 0] + (2.0 / 3.0) * rho_in * U_in
    f[5, s, 0] = f[7, s, 0] + (1.0 / 6.0) * rho_in * U_in \
        - 0.5 * (f[2, s, 0] - f[4, s, 0])
    f[8, s, 0] = f[6, s, 0] + (1.0 / 6.0) * rho_in * U_in \
        + 0.5 * (f[2, s, 0] - f[4, s, 0])
