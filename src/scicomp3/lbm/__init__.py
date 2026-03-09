"""Lattice Boltzmann Method (LBM) for 2D incompressible flow.

Implements the D2Q9 model with BGK collision operator for solving
the incompressible Navier-Stokes equations.
"""

from .d2q9 import D2Q9
from .solver import LBMSolver, LBMResult
from .obstacles import cylinder_mask, rectangle_mask

__all__ = [
    "D2Q9",
    "LBMSolver",
    "LBMResult",
    "cylinder_mask",
    "rectangle_mask",
]
