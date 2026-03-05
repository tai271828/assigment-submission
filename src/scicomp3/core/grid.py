"""Grid definitions for spatial discretization."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Grid1D:
    """1D uniform grid for spatial discretization.

    Attributes:
        N: Number of grid intervals
        L: Domain length
        dx: Grid spacing (computed as L/N)
        x: Array of grid point coordinates
    """

    N: int
    L: float = 1.0
    dx: float = field(init=False)
    x: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.dx = self.L / self.N
        self.x = np.arange(self.N + 1) * self.dx

    @property
    def shape(self) -> tuple:
        """Return the shape of the grid."""
        return (self.N,)


@dataclass
class Grid2D:
    """2D uniform grid for spatial discretization.

    Grid points: i,j in (0, 1, ..., N), giving N+1 points in each direction.

    Attributes:
        N: Number of grid intervals in each direction
        L: Domain length (square domain)
        dx: Grid spacing (computed as L/N)
        x: 1D array of x-coordinates
        y: 1D array of y-coordinates
        X: 2D meshgrid of x-coordinates
        Y: 2D meshgrid of y-coordinates
    """

    N: int
    L: float = 1.0
    dx: float = field(init=False)
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N + 1)
        self.y = np.linspace(0, self.L, self.N + 1)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

    @property
    def shape(self) -> tuple:
        """Return the shape of the grid."""
        return (self.N + 1, self.N + 1)


def get_neighbours(N, i, j):
    """
    Helper function.
    Returns coordinates of the four neighbours of the given point (i,j)
    for a grid with size N.

    Wraps around for the j coordinate, and clamps for the i coordinate
    """
    i_min = (i - 1) % (N + 1)
    i_plus = (i + 1) % (N + 1)
    j_min = max(j - 1, 0)
    j_plus = min(j + 1, N)
    return [(i_min, j), (i_plus, j), (i, j_min), (i, j_plus)]
