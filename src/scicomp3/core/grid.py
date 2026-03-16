"""Grid definitions for spatial discretization."""

from dataclasses import dataclass, field
import numpy as np

from warnings import warn


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
    """2D uniform grid for spatial discretisation on a rectangular domain.

    Grid points: i in (0, 1, ..., Nx), j in (0, 1, ..., Ny).
    For a square domain, set Ny=Nx and Ly=Lx (or omit them).

    Attributes:
        Nx: Number of grid intervals in x
        Ny: Number of grid intervals in y (defaults to Nx for square domains)
        Lx: Domain length in x
        Ly: Domain length in y (defaults to Lx for square domains)
        dx: Grid spacing in x (Lx / Nx)
        dy: Grid spacing in y (Ly / Ny)
        x: 1D array of x-coordinates
        y: 1D array of y-coordinates
        X: 2D meshgrid of x-coordinates, shape (Nx+1, Ny+1)
        Y: 2D meshgrid of y-coordinates, shape (Nx+1, Ny+1)
    """

    Nx: int = None
    Lx: float = None
    Ny: int = None
    Ly: float = None
    # Legacy parameters
    N: int = field(default=None, repr=False)
    L: float = field(default=None, repr=False)

    dx: float = field(init=False)
    dy: float = field(init=False)
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Resolve legacy parameters
        if self.N is not None:
            warn(
                "N is deprecated, use Nx (and Ny) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.Nx is None:
                self.Nx = self.N
            if self.Ny is None:
                self.Ny = self.N
        if self.L is not None:
            warn(
                "L is deprecated, use Lx (and Ly) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.Lx is None:
                self.Lx = self.L
            if self.Ly is None:
                self.Ly = self.L

        # Validate
        if self.Nx is None:
            raise ValueError("Nx must be provided.")
        if self.Lx is None:
            raise ValueError("Lx must be provided.")

        # Apply defaults for square domain
        if self.Ny is None:
            self.Ny = self.Nx
        if self.Ly is None:
            self.Ly = self.Lx

        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.x = np.linspace(0, self.Lx, self.Nx + 1)
        self.y = np.linspace(0, self.Ly, self.Ny + 1)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the grid."""
        return (self.Nx + 1, self.Ny + 1)


def get_neighbours(N, i, j):
    """
    Helper function.
    Returns coordinates of the four neighbours of the given point (i,j)
    for a square grid with shape N+1 x N+1.

    Wraps around for the j coordinate, and clamps for the i coordinate
    """
    i_min = (i - 1) % (N + 1)
    i_plus = (i + 1) % (N + 1)
    j_min = max(j - 1, 0)
    j_plus = min(j + 1, N)
    return [(i_min, j), (i_plus, j), (i, j_min), (i, j_plus)]
