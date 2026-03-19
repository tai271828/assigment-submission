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
from numba import njit

from ..core.config import KVSConfig
from ..core.result import KVSResult

from ..ode.solver import solve_ivp

# =============================================================================
# D2Q9 Lattice Definition (for LBM solver)
# =============================================================================
#
#   6  2  5        Lattice velocities c_i (i = 0..8):
#    \ | /           0: rest          (0, 0)
#   3--0--1          1-4: axis-aligned  (±1,0), (0,±1)
#    / | \           5-8: diagonals     (±1,±1)
#   7  4  8

_LB_C = np.array(
    [
        [0, 0],  # 0  — rest
        [1, 0],  # 1  — east
        [0, 1],  # 2  — north
        [-1, 0],  # 3  — west
        [0, -1],  # 4  — south
        [1, 1],  # 5  — north-east
        [-1, 1],  # 6  — north-west
        [-1, -1],  # 7  — south-west
        [1, -1],  # 8  — south-east
    ]
)

_LB_W = np.array(
    [
        4 / 9,  # rest
        1 / 9,
        1 / 9,
        1 / 9,
        1 / 9,  # axis-aligned
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,  # diagonals
    ]
)

# Opposite direction index for each i (used in bounce-back)
_LB_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

_LB_NDIR = 9


@njit
def _lb_equilibrium(rho, ux, uy):
    """
    Compute the D2Q9 equilibrium distribution f^eq (BGK).

        f_i^eq = w_i * rho * (1 + c_i·u/cs² + (c_i·u)²/(2·cs⁴) - u·u/(2·cs²))

    where cs² = 1/3 (lattice speed of sound squared).
    """
    Nx, Ny = rho.shape
    feq = np.zeros((Nx, Ny, 9))
    usqr = ux**2 + uy**2

    for i in range(9):
        cu = _LB_C[i, 0] * ux + _LB_C[i, 1] * uy
        feq[:, :, i] = _LB_W[i] * rho * (
            1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usqr
        )
    return feq


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


def _solve_kvs_lb(
    config: KVSConfig,
    n_steps: int,
    plot_every: int = 100,
    post_step=None,
    U_lb: float = 0.1,
    **kwargs,
) -> KVSResult:
    """LBM implementation of the KVS solver using D2Q9 lattice and BGK collision.

    Based on Gabor's lbm_karman-Drag_lift.py.  Uses the collide-then-stream
    pattern with bounce-back for no-slip walls and Zou-He inlet BC.

    Algorithm per timestep:
      1. Compute macroscopic quantities (density, velocity) from distributions
      2. Collision step  — relax f toward local equilibrium (BGK)
      3. Bounce-back    — reflect populations at solid nodes (cylinder + walls)
      4. Streaming step — propagate f_i along lattice velocity c_i
      5. Boundary conditions — Zou-He inlet, zero-gradient outlet

    Args:
        config:     KVSConfig with physical and grid parameters.
        n_steps:    Number of LBM timesteps.
        plot_every: Save a snapshot every this many steps. 0 to disable.
        post_step:  Optional callback (unused — LBM handles BCs internally).
        U_lb:       Inlet velocity in lattice units (default: 0.1, keep ≪ 1).
        **kwargs:   Passed through (unused).

    Returns:
        KVSResult with final velocity field, snapshots, config, and method.
    """
    # ------------------------------------------------------------------
    # Grid and parameter mapping (physical -> lattice)
    # ------------------------------------------------------------------
    Nx_lb, Ny_lb = config.grid.shape  # (Nx+1, Ny+1) lattice sites

    if abs(config.dx - config.dy) > 1e-10 * config.dx:
        raise ValueError(
            f"LBM requires dx == dy; got dx={config.dx}, dy={config.dy}. "
            "Adjust Nx/Ny so that Lx/Nx == Ly/Ny."
        )

    D_lb = 2 * config.radius / config.dx  # cylinder diameter in lattice units
    nu_lb = U_lb * D_lb / config.Re  # kinematic viscosity in lattice units
    tau = 3.0 * nu_lb + 0.5  # BGK relaxation time

    # Velocity conversion factor: u_physical = u_lattice * vel_scale
    vel_scale = config.U_inlet / U_lb

    print(f"LBM parameters:")
    print(f"  Grid:    {Nx_lb} x {Ny_lb}")
    print(f"  D_lb={D_lb:.1f},  U_lb={U_lb},  nu_lb={nu_lb:.6f},  tau={tau:.4f}")

    # ------------------------------------------------------------------
    # Solid mask: cylinder + top/bottom channel walls
    # ------------------------------------------------------------------
    obstacle = config.cylinder_mask  # shape (Nx+1, Ny+1)
    wall = np.zeros((Nx_lb, Ny_lb), dtype=bool)
    wall[:, 0] = True  # bottom wall
    wall[:, -1] = True  # top wall
    solid = obstacle | wall

    # ------------------------------------------------------------------
    # Initialization: uniform flow at lattice inlet velocity
    # ------------------------------------------------------------------
    rho = np.ones((Nx_lb, Ny_lb))
    ux = np.full((Nx_lb, Ny_lb), U_lb)
    uy = np.zeros((Nx_lb, Ny_lb))

    # Small transverse perturbation to break symmetry and trigger shedding
    j_arr = np.arange(Ny_lb)[None, :]  # broadcast over x
    uy += 0.001 * U_lb * np.sin(2.0 * np.pi * j_arr / Ny_lb)

    # Zero velocity inside solid
    ux[solid] = 0.0
    uy[solid] = 0.0

    # Initialize distributions to equilibrium
    f = _lb_equilibrium(rho, ux, uy)

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    snapshots = []
    drag_history = np.zeros(n_steps)
    lift_history = np.zeros(n_steps)

    print(f"\nRunning {n_steps} LBM timesteps ...")

    for step in range(1, n_steps + 1):

        # --- Macroscopic quantities: rho = Σ f_i, rho·u = Σ c_i · f_i ---
        rho = np.sum(f, axis=2)
        ux = np.sum(f * _LB_C[:, 0], axis=2) / rho
        uy = np.sum(f * _LB_C[:, 1], axis=2) / rho

        # --- Collision step (BGK single-relaxation-time) ---
        feq = _lb_equilibrium(rho, ux, uy)
        f_out = f - (f - feq) / tau

        # --- Bounce-back on solid nodes + momentum exchange on cylinder ---
        #     Momentum-exchange: F = Σ c_i * (f_out_i + f_out_opp_i)
        Fx, Fy = 0.0, 0.0
        for i in range(_LB_NDIR):
            Fx += _LB_C[i, 0] * np.sum(
                f_out[obstacle, i] + f_out[obstacle, _LB_OPP[i]]
            )
            Fy += _LB_C[i, 1] * np.sum(
                f_out[obstacle, i] + f_out[obstacle, _LB_OPP[i]]
            )
            f_out[solid, i] = f[solid, _LB_OPP[i]]
        drag_history[step - 1] = Fx
        lift_history[step - 1] = Fy

        # --- Streaming step: shift each f_i by its lattice velocity c_i ---
        for i in range(_LB_NDIR):
            f[:, :, i] = np.roll(f_out[:, :, i], shift=_LB_C[i, 0], axis=0)
            f[:, :, i] = np.roll(f[:, :, i], shift=_LB_C[i, 1], axis=1)

        # --- Outlet BC (zero-gradient / open) ---
        f[-1, :, :] = f[-2, :, :]

        # --- Inlet BC (Zou-He, fixed velocity: ux=U_lb, uy=0) ---
        rho_in = (
            (f[0, :, 0] + f[0, :, 2] + f[0, :, 4])
            + 2.0 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])
        ) / (1.0 - U_lb)

        f[0, :, 1] = f[0, :, 3] + (2.0 / 3.0) * rho_in * U_lb
        f[0, :, 5] = (
            f[0, :, 7]
            - 0.5 * (f[0, :, 2] - f[0, :, 4])
            + (1.0 / 6.0) * rho_in * U_lb
        )
        f[0, :, 8] = (
            f[0, :, 6]
            + 0.5 * (f[0, :, 2] - f[0, :, 4])
            + (1.0 / 6.0) * rho_in * U_lb
        )

        # --- Save snapshot (converted to physical velocity) ---
        if plot_every > 0 and step % plot_every == 0:
            u_snapshot = np.array([ux * vel_scale, uy * vel_scale])
            snapshots.append(u_snapshot)

        if step % 1000 == 0:
            avg_rho = np.mean(rho[~solid])
            print(f"  Step {step:>6d}/{n_steps}  |  avg density = {avg_rho:.6f}")

    # ------------------------------------------------------------------
    # Return result
    # ------------------------------------------------------------------
    u_final = np.array([ux * vel_scale, uy * vel_scale])

    print(f"KVS LBM done. {len(snapshots)} snapshots saved.")

    return KVSResult(
        u=u_final,
        snapshots=snapshots,
        config=config,
        method="lb",
    )


METHODS = {
    "fd": _solve_kvs_fd,
    "fe": _solve_kvs_fe,
    "lb": _solve_kvs_lb,
}
