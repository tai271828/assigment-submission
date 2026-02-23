"""
scicomp3: Scientific Computing Package for Assignment Sets 1 & 2

Solvers for wave equation (1D), diffusion (2D), Laplace BVP,
Diffusion-Limited Aggregation (PDE and Monte Carlo), and the
Gray-Scott reaction-diffusion system.
"""

__version__ = "0.2.1"

from .core.grid import Grid1D
from .core.result import ODEResult, BVPResult
from .ode.solver import solve_ivp
from .ode.methods import METHODS
from .bvp.solver import solve_bvp
from .pde.wave import wave1d_rhs
from .pde.diffusion import diffusion2d_rhs
from .pde.gray_scott import gray_scott_rhs, gray_scott_initial_conditions

__all__ = [
    "Grid1D",
    "ODEResult",
    "BVPResult",
    "solve_ivp",
    "METHODS",
    "solve_bvp",
    "wave1d_rhs",
    "diffusion2d_rhs",
    "gray_scott_rhs",
    "gray_scott_initial_conditions",
]
