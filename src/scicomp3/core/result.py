"""Result containers for solvers."""

from dataclasses import dataclass
import numpy as np

from .config import KVSConfig


@dataclass
class ODEResult:
    """Container for ODE/IVP solver results.

    Follows scipy conventions for result objects.

    Attributes:
        t: Time points array
        y: Solution array at each time point
        success: Whether the integration completed
        message: Description of termination
        nfev: Number of function evaluations
    """

    t: np.ndarray
    y: np.ndarray
    success: bool = True
    message: str = ""
    nfev: int = 0


@dataclass
class BVPResult:
    """Container for BVP iterative solver results.

    Attributes:
        y: Final solution array
        converged: Whether convergence criterion was met
        n_iter: Number of iterations performed
        delta_history: Convergence measure delta at each iteration
    """

    y: np.ndarray
    converged: bool = True
    n_iter: int = 0
    delta_history: np.ndarray = None


@dataclass
class DLASORResult:
    """Container for DLA results by SOR.

    Attributes:
        y: Final solution array
        growth_mask: Final mask array marking the grown aggregate
        bvp_iters_per_step: BVP iteration count for each growth step
    """

    y: np.ndarray
    growth_mask: np.ndarray
    bvp_iters_per_step: np.ndarray = None


@dataclass
class DLAMCResult:
    """Container for DLA results by MC (Monte Carlo).

    Attributes:
        growth_mask: Final mask array marking the grown aggregate
    """

    growth_mask: np.ndarray


@dataclass
class DLAMCResultLegacy:
    """Legacy ontainer for DLA results by MC (Monte Carlo).

    Attributes:
        walkers_mask: Final grid with walkers
        growth_mask: Final mask array marking the grown aggregate
    """

    walkers_mask: np.ndarray
    growth_mask: np.ndarray


@dataclass
class KVSResult:
    """Result container for a Kármán Vortex Street simulation.

    Attributes:
        u:          Final velocity field, shape (Nx+1, Ny+1).
        snapshots:  List of velocity field snapshots taken every plot_every
                    steps.
        config:     KVSConfig used for the simulation.
        method:     Solver method used, e.g. 'fd', 'fe', 'lb'.
    """

    u: np.ndarray
    snapshots: list
    config: KVSConfig
    method: str


def find_y(res: ODEResult, t):
    """Compute the y-value corresponding to the given t value"""
    for i in range(len(res.t)):
        if res.t[i] == t:
            return res.y[i]
    raise ValueError
