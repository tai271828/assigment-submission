"""LBM solver for 2D incompressible Navier-Stokes using BGK collision.

Implements the standard collide-stream algorithm on a D2Q9 lattice:
  1. Collision (BGK relaxation toward equilibrium)
  2. Streaming (propagation along lattice velocities)
  3. Boundary conditions (bounce-back, equilibrium inlet, outflow)

Key stability considerations:
  - tau must be > 0.5 (tau = 0.5 gives zero viscosity).
  - The lattice Mach number Ma = U_lb / cs should be small (< 0.1).
  - An initial velocity ramp avoids transient instabilities.

Physical-to-lattice unit conversion:
  - Choose lattice resolution so that the cylinder diameter D maps to
    D_lb lattice nodes.
  - Lattice spacing dx_lb = 1, time step dt_lb = 1.
  - Given Re = U_phys * D_phys / nu_phys, pick U_lb (small, e.g. 0.04-0.1)
    and compute nu_lb = U_lb * D_lb / Re.
  - Relaxation time tau = nu_lb / cs² + 0.5 = 3*nu_lb + 0.5.
"""

from dataclasses import dataclass
import numpy as np

from .d2q9 import D2Q9


@dataclass
class LBMResult:
    """Container for LBM simulation results.

    Attributes:
        rho: Final density field, shape (Nx, Ny).
        ux: Final x-velocity field, shape (Nx, Ny).
        uy: Final y-velocity field, shape (Nx, Ny).
        obstacle: Obstacle mask.
        history: List of (step, rho, ux, uy) snapshots at saved intervals.
    """

    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    obstacle: np.ndarray
    history: list


class LBMSolver:
    """2D Lattice Boltzmann solver with D2Q9 model and BGK collision.

    Uses equilibrium-based inlet BC (more stable than Zou-He for low tau)
    and pre-streaming bounce-back for obstacles and walls.

    Parameters:
        Nx: Lattice nodes in x (streamwise).
        Ny: Lattice nodes in y (cross-stream).
        tau: BGK relaxation time. Related to viscosity by
             nu = cs² * (tau - 0.5) = (tau - 0.5) / 3.
        obstacle: Boolean mask of shape (Nx, Ny), True at solid nodes.
        u_inlet: Peak inlet velocity magnitude (lattice units).
                 Should satisfy Ma = u_inlet * sqrt(3) << 1 for accuracy.
    """

    def __init__(self, Nx, Ny, tau, obstacle, u_inlet=0.04):
        self.Nx = Nx
        self.Ny = Ny
        self.tau = tau
        self.omega = 1.0 / tau
        self.obstacle = obstacle
        self.u_inlet = u_inlet
        self._current_step = 0
        self._ramp_steps = 0  # set in solve() if ramping is used

        # Precompute parabolic inlet profile: u(y) = 4*U*y*(H-y)/(H)²
        y = np.arange(Ny)
        H = Ny - 1
        self.u_inlet_profile = 4.0 * u_inlet * y * (H - y) / H**2

        # Initialize fields: uniform density, parabolic velocity everywhere
        self.rho = np.ones((Nx, Ny))
        self.ux = np.zeros((Nx, Ny))
        self.uy = np.zeros((Nx, Ny))
        for i in range(Nx):
            self.ux[i, :] = self.u_inlet_profile

        # Zero velocity inside obstacle
        self.ux[obstacle] = 0.0
        self.uy[obstacle] = 0.0

        # Initialize distribution to equilibrium
        self.f = D2Q9.equilibrium(self.rho, self.ux, self.uy)

        # Pre-compute fluid mask (interior nodes that are not obstacle)
        self._fluid = ~obstacle

    @property
    def nu(self):
        """Kinematic viscosity in lattice units."""
        return D2Q9.cs2 * (self.tau - 0.5)

    def _get_ramp_factor(self):
        """Smooth velocity ramp factor: 0 at start -> 1 after ramp_steps."""
        if self._ramp_steps <= 0:
            return 1.0
        t = min(self._current_step / self._ramp_steps, 1.0)
        # Smooth ramp: 0.5*(1 - cos(pi*t))
        return 0.5 * (1.0 - np.cos(np.pi * t))

    def _collide(self):
        """BGK collision: relax toward equilibrium."""
        feq = D2Q9.equilibrium(self.rho, self.ux, self.uy)
        self.f += self.omega * (feq - self.f)

    def _stream(self):
        """Stream distributions along lattice velocities using np.roll."""
        e = D2Q9.e
        f_new = np.empty_like(self.f)
        for i in range(9):
            f_new[i] = np.roll(
                np.roll(self.f[i], e[i, 0], axis=0),
                e[i, 1], axis=1,
            )
        self.f = f_new

    def _bounce_back_obstacle(self):
        """Post-streaming bounce-back on obstacle nodes.

        After streaming, distributions that arrived at obstacle nodes
        are reflected to opposite directions. They will stream back to
        the fluid in the next time step.
        """
        opp = D2Q9.opposite
        f_obs = self.f[:, self.obstacle].copy()
        for i in range(9):
            self.f[i][self.obstacle] = f_obs[opp[i]]

    def _apply_wall_bc(self):
        """No-slip bounce-back on top (y=Ny-1) and bottom (y=0) walls.

        After streaming, south-going distributions at y=0 are reflected
        north, and north-going distributions at y=Ny-1 are reflected south.
        """
        f = self.f

        # Bottom wall (y=0): reflect south-going -> north-going
        f[2, :, 0] = f[4, :, 0]   # 4(south) -> 2(north)
        f[5, :, 0] = f[7, :, 0]   # 7(south-west) -> 5(north-east)
        f[6, :, 0] = f[8, :, 0]   # 8(south-east) -> 6(north-west)

        # Top wall (y=Ny-1): reflect north-going -> south-going
        f[4, :, -1] = f[2, :, -1]  # 2(north) -> 4(south)
        f[7, :, -1] = f[5, :, -1]  # 5(north-east) -> 7(south-west)
        f[8, :, -1] = f[6, :, -1]  # 6(north-west) -> 8(south-east)

    def _apply_inlet_bc(self):
        """Equilibrium inlet BC at x=0 with prescribed parabolic velocity.

        More stable than Zou-He for low tau values. Sets the full
        distribution at inlet nodes to the equilibrium for the
        prescribed velocity and current local density.
        """
        ramp = self._get_ramp_factor()
        ux_in = self.u_inlet_profile * ramp
        uy_in = np.zeros(self.Ny)

        # Use local density (more stable than fixing rho=1)
        rho_in = self.rho[0, :]

        feq_in = D2Q9.equilibrium(
            rho_in[np.newaxis, :],
            ux_in[np.newaxis, :],
            uy_in[np.newaxis, :],
        )
        self.f[:, 0, :] = feq_in[:, 0, :]

    def _apply_outlet_bc(self):
        """Zero-gradient (extrapolation) outflow at x=Nx-1.

        Copy distributions from the second-to-last column.
        """
        self.f[:, -1, :] = self.f[:, -2, :]

    def _compute_macroscopic(self):
        """Compute density and velocity from distribution functions."""
        e = D2Q9.e
        self.rho = np.sum(self.f, axis=0)

        # Avoid division by zero
        rho_safe = np.where(self.rho > 1e-10, self.rho, 1.0)

        self.ux = np.zeros_like(self.rho)
        self.uy = np.zeros_like(self.rho)
        for i in range(9):
            self.ux += e[i, 0] * self.f[i]
            self.uy += e[i, 1] * self.f[i]
        self.ux /= rho_safe
        self.uy /= rho_safe

        # Zero velocity on obstacle nodes
        self.ux[self.obstacle] = 0.0
        self.uy[self.obstacle] = 0.0

    def step(self):
        """Perform one LBM time step: collide -> stream -> BCs -> macroscopic."""
        self._current_step += 1

        self._collide()
        self._stream()

        # Boundary conditions (order matters)
        self._apply_wall_bc()
        self._bounce_back_obstacle()
        self._apply_inlet_bc()
        self._apply_outlet_bc()

        self._compute_macroscopic()

    def solve(self, n_steps, save_every=None, callback=None, ramp_steps=0):
        """Run the LBM simulation for n_steps.

        Parameters:
            n_steps: Number of time steps.
            save_every: Save snapshots every N steps. None = no saving.
            callback: Optional callable(step, solver) called each step.
            ramp_steps: Number of steps to ramp inlet velocity from 0 to full.
                        Helps avoid initial transient instabilities.

        Returns:
            LBMResult with final state and history.
        """
        self._ramp_steps = ramp_steps
        history = []

        for step_num in range(n_steps):
            self.step()

            if save_every and self._current_step % save_every == 0:
                history.append((
                    self._current_step,
                    self.rho.copy(),
                    self.ux.copy(),
                    self.uy.copy(),
                ))

            if callback is not None:
                callback(self._current_step, self)

        return LBMResult(
            rho=self.rho.copy(),
            ux=self.ux.copy(),
            uy=self.uy.copy(),
            obstacle=self.obstacle.copy(),
            history=history,
        )


def compute_re(u_inlet, D_lb, nu_lb):
    """Compute Reynolds number from lattice quantities.

    Re = U_max * D / nu
    """
    return u_inlet * D_lb / nu_lb


def tau_from_re(Re, u_inlet, D_lb):
    """Compute relaxation time tau from desired Reynolds number.

    nu_lb = u_inlet * D_lb / Re
    tau = 3 * nu_lb + 0.5
    """
    nu_lb = u_inlet * D_lb / Re
    return 3.0 * nu_lb + 0.5
