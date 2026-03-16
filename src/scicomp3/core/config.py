"""
Dataclasses that each contain the configuration for a simulation
"""

import numpy as np
from dataclasses import dataclass, field

from .grid import Grid2D
from ..objects.shapes import construct_cylinder


@dataclass
class KVSConfig:
    """
    Physical and discretisation parameters for the Kármán Vortex Street.

    Physical parameters default to the assignment spec (Figure 4).
    Override them to experiment with different setups. The cylinder center
    defaults to (0.2, 0.2), consistent with the spec (0.15 + radius).

    Args:
        Lx:       Domain length in x (default: 2.2 m)
        Ly:       Domain height in y (default: 0.41 m)
        cx:       Cylinder center x (default: 0.2 m)
        cy:       Cylinder center y (default: 0.2 m)
        radius:   Cylinder radius (default: 0.05 m)
        Re:       Reynolds number (default: 150)
        U_inlet:  Inlet velocity (default: 0.1 m/s)
        Nx:       Grid intervals in x (default: 220)
        Ny:       Grid intervals in y (default: 41)

    Derived attributes (not passed to constructor):
        grid:           Grid2D instance for the rectangular domain.
        cylinder_mask:  Boolean array of shape (Nx+1, Ny+1), True at grid
                        points inside the cylinder.

    Derived properties:
        nu:   Kinematic viscosity, computed as U_inlet * D / Re [m²/s]
        dx:   Grid spacing in x [m]
        dy:   Grid spacing in y [m]
    """

    Lx: float = 2.2
    Ly: float = 0.41
    cx: float = 0.2
    cy: float = 0.2
    radius: float = 0.05
    Re: float = 150.0
    U_inlet: float = 0.1
    Nx: int = 220
    Ny: int = 41

    # Derived — computed in __post_init__
    grid: Grid2D = field(init=False, repr=False)
    cylinder_mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.grid = Grid2D(Nx=self.Nx, Lx=self.Lx, Ny=self.Ny, Ly=self.Ly)
        self.cylinder_mask = construct_cylinder(
            self.grid, self.cx, self.cy, self.radius
        )

    @property
    def nu(self) -> float:
        """Kinematic viscosity derived from Re = U * D / nu."""
        return self.U_inlet * (2 * self.radius) / self.Re

    @property
    def dx(self) -> float:
        return self.grid.dx

    @property
    def dy(self) -> float:
        return self.grid.dy
