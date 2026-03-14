"""
Generating arrays of coordinates for geometric descriptions
"""

import numpy as np

from ..core.grid import Grid2D


def construct_rectangle(xmin: int, xmax: int, ymin: int, ymax: int):
    """
    Returns an np array with the coordinates of the given rectangle
    e.g. for (3,5,4,5):
         [(3,4), (4,4), (5,4), (3,5), (4,5), (5,5)]
    """
    xs = np.arange(xmin, xmax + 1)
    ys = np.arange(ymin, ymax + 1)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


def construct_circle(x_center: int, y_center: int, radius: float):
    """
    Returns an np array with the integer coordinates that fall within
    the given circle (including the boundary).
    """
    r = int(np.ceil(radius))

    # bounding square around the circle
    xs = np.arange(x_center - r, x_center + r + 1)
    ys = np.arange(y_center - r, y_center + r + 1)

    X, Y = np.meshgrid(xs, ys)

    # distance condition
    mask = (X - x_center) ** 2 + (Y - y_center) ** 2 <= radius**2

    return np.column_stack([X[mask], Y[mask]])


def construct_cylinder(
    grid: Grid2D, x_center: float, y_center: float, radius: float
) -> np.ndarray:
    """
    Return a boolean mask of shape (Nx+1, Ny+1) that is True at grid points
    inside or on the boundary of the circle with given physical center and radius.

    Args:
        grid:     Grid2D instance defining the spatial discretisation.
        x_center: Physical x coordinate of the circle center [m].
        y_center: Physical y coordinate of the circle center [m].
        radius:   Physical radius of the circle [m].

    Returns:
        mask: Boolean array of shape (Nx+1, Ny+1).
    """
    return (grid.X - x_center) ** 2 + (grid.Y - y_center) ** 2 <= radius**2
