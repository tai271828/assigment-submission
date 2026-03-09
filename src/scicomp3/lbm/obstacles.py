"""Obstacle geometry definitions for LBM simulations."""

import numpy as np


def cylinder_mask(Nx, Ny, cx, cy, r):
    """Create a boolean mask for a circular cylinder obstacle.

    Parameters:
        Nx: Number of lattice nodes in x-direction.
        Ny: Number of lattice nodes in y-direction.
        cx: x-coordinate of cylinder center (in lattice units).
        cy: y-coordinate of cylinder center (in lattice units).
        r: Radius of cylinder (in lattice units).

    Returns:
        mask: Boolean array of shape (Nx, Ny), True inside obstacle.
    """
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r**2


def rectangle_mask(Nx, Ny, x_min, x_max, y_min, y_max):
    """Create a boolean mask for a rectangular obstacle.

    Parameters:
        Nx: Number of lattice nodes in x-direction.
        Ny: Number of lattice nodes in y-direction.
        x_min, x_max: x-extent of rectangle (in lattice units).
        y_min, y_max: y-extent of rectangle (in lattice units).

    Returns:
        mask: Boolean array of shape (Nx, Ny), True inside obstacle.
    """
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
