"""
Numerical solvers for BVP the Kármán Vortex Street

- _solve_kvs_fd:            Finite Difference solver for the Kármán Vortex Street
- _solve_kvs_fe:            Finite Element solver for the Kármán Vortex Street
- _solve_kvs_lb:            Lattice Bolzmann solver for the Kármán Vortex Street

All solver functions have signature:
    solver(config, n_steps, plot_every, post_step, **kwargs) -> KVSResult

The METHODS registry maps string keys to solver functions
"""

import numpy as np

from ..core.config import KVSConfig
from ..core.result import KVSResult

from ..ode.solver import solve_ivp


def _apply_fd_bc(config: KVSConfig, u: np.ndarray) -> np.ndarray:
    """Apply KVS boundary conditions for the FD solver.

    Args:
        config: KVSConfig with grid and cylinder mask.
        u: Velocity field of shape (2, Nx+1, Ny+1), where u[0]=ux, u[1]=uy.
    """
    u[:, :, 0] = 0  # bottom wall
    u[:, :, -1] = 0  # top wall
    u[:, config.cylinder_mask] = 0  # cylinder (no-slip)
    u[0, 0, :] = config.U_inlet  # inlet: ux = U_inlet
    u[1, 0, :] = 0  # inlet: uy = 0
    u[:, -1, :] = u[:, -2, :]  # outlet (zero-gradient)
    return u


def _rhs_navier_stokes(t, u, config):
    """
    Evaluate the Navier-Stokes RHS at the current velocity field u.
    u has shape (2, Nx+1, Ny+1): u[0] = ux, u[1] = uy
    """
    ux, uy = u[0], u[1]
    dx, dy, nu = config.dx, config.dy, config.nu

    # Diffusion: nu * nabla^2 u
    d2ux = (np.roll(ux, -1, axis=0) - 2 * ux + np.roll(ux, 1, axis=0)) / dx**2 + (
        np.roll(ux, -1, axis=1) - 2 * ux + np.roll(ux, 1, axis=1)
    ) / dy**2
    d2uy = (np.roll(uy, -1, axis=0) - 2 * uy + np.roll(uy, 1, axis=0)) / dx**2 + (
        np.roll(uy, -1, axis=1) - 2 * uy + np.roll(uy, 1, axis=1)
    ) / dy**2

    # Advection: (u·nabla) u
    dux_dx = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2 * dx)
    dux_dy = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2 * dy)
    duy_dx = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2 * dx)
    duy_dy = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2 * dy)

    dux_dt = nu * d2ux - (ux * dux_dx + uy * dux_dy)
    duy_dt = nu * d2uy - (ux * duy_dx + uy * duy_dy)

    return np.array([dux_dt, duy_dt])


def _solve_kvs_fd(
    config: KVSConfig,
    n_steps: int,
    plot_every: int = 100,
    post_step=None,
    dt: float = 1e-3,
    **kwargs,
) -> KVSResult:
    """FD implementation of the KVS solver using forward Euler time-stepping.

    Solves the incompressible Navier-Stokes equations using finite differences.
    The diffusion and advection terms are evaluated explicitly; note that this
    omits the pressure projection step, so strict incompressibility is not
    enforced. This is a simplified starting point — add a pressure Poisson
    solve between the advection and correction steps for a fully correct solver.

    Args:
        config:     KVSConfig with physical and grid parameters.
        n_steps:    Number of time steps.
        plot_every: Save a snapshot every this many steps. 0 to disable.
        post_step:  Optional callback f(t, u) -> u applied after each step,
                    in addition to the KVS boundary conditions.
        dt:         Time step size (default: 1e-3).
        **kwargs:   Passed through (unused).

    Returns:
        KVSResult with final velocity field, snapshots, config, and method.
    """

    def _bc_and_post(t, u):
        u = _apply_fd_bc(config, u)
        if post_step is not None:
            u = post_step(t, u)
        return u

    result = solve_ivp(
        fun=_rhs_navier_stokes,
        t_span=(0, n_steps * dt),
        y0=np.zeros((2, config.grid.Nx + 1, config.grid.Ny + 1)),
        method="forward_euler",
        dt=dt,
        args=(config,),
        post_step=_bc_and_post,
        save_interval=plot_every if plot_every > 0 else n_steps,
    )

    snapshots = [
        frame.reshape(2, config.grid.Nx + 1, config.grid.Ny + 1) for frame in result.y
    ]
    # Progress is handled by solve_ivp internally — add a print here if needed
    print(f"KVS FD done. {len(snapshots)} snapshots saved.")

    return KVSResult(
        u=result.y[-1],
        snapshots=snapshots,
        config=config,
        method="fd",
    )


def _solve_kvs_fe(**kwargs):
    # Emma will implement this later
    raise NotImplementedError


def _solve_kvs_lb(**kwargs):
    # Tai will implement this later
    raise NotImplementedError


METHODS = {
    "fd": _solve_kvs_fd,
    "fe": _solve_kvs_fe,
    "lb": _solve_kvs_lb,
}
