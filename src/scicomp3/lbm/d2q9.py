"""D2Q9 lattice model constants and equilibrium distribution."""

import numpy as np


class D2Q9:
    """D2Q9 lattice constants for 2D Lattice Boltzmann.

    Velocity directions:
        6 2 5
         \\|/
        3-0-1
         /|\\
        7 4 8

    Attributes:
        e: Lattice velocity vectors, shape (9, 2).
        w: Lattice weights, shape (9,).
        cs2: Speed of sound squared (1/3).
        opposite: Index of opposite direction for each velocity.
    """

    # Lattice velocities (x, y)
    e = np.array([
        [0, 0],    # 0: rest
        [1, 0],    # 1: east
        [0, 1],    # 2: north
        [-1, 0],   # 3: west
        [0, -1],   # 4: south
        [1, 1],    # 5: north-east
        [-1, 1],   # 6: north-west
        [-1, -1],  # 7: south-west
        [1, -1],   # 8: south-east
    ])

    # Weights
    w = np.array([
        4 / 9,                          # rest
        1 / 9, 1 / 9, 1 / 9, 1 / 9,    # cardinal
        1 / 36, 1 / 36, 1 / 36, 1 / 36, # diagonal
    ])

    # Opposite direction indices (for bounce-back)
    opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    # Speed of sound squared
    cs2 = 1.0 / 3.0

    @staticmethod
    def equilibrium(rho, ux, uy):
        """Compute equilibrium distribution function f^eq.

        f_i^eq = w_i * rho * (1 + e_i·u/cs² + (e_i·u)²/(2*cs⁴) - u²/(2*cs²))

        Parameters:
            rho: Density field, shape (Nx, Ny).
            ux: x-velocity field, shape (Nx, Ny).
            uy: y-velocity field, shape (Nx, Ny).

        Returns:
            feq: Equilibrium distributions, shape (9, Nx, Ny).
        """
        e = D2Q9.e
        w = D2Q9.w
        cs2 = D2Q9.cs2

        usq = ux**2 + uy**2  # |u|²

        feq = np.zeros((9,) + rho.shape)
        for i in range(9):
            eu = e[i, 0] * ux + e[i, 1] * uy  # e_i · u
            feq[i] = w[i] * rho * (
                1.0 + eu / cs2 + 0.5 * eu**2 / cs2**2 - 0.5 * usq / cs2
            )
        return feq
