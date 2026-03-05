"""Result containers for solvers."""

from dataclasses import dataclass
import numpy as np


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
    """

    y: np.ndarray
    growth_mask: np.ndarray


@dataclass
class DLAMCResult:
    """Container for DLA results by MC (Monte Carlo).

    Attributes:
        growth_mask: Final mask array marking the grown aggregate
    """

    walkers_mask: np.ndarray
    growth_mask: np.ndarray


def find_y(res: ODEResult, t):
    """Compute the y-value corresponding to the given t value"""
    for i in range(len(res.t)):
        if res.t[i] == t:
            return res.y[i]
    raise ValueError
