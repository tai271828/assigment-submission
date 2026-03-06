"""Numba-accelerated SOR iterations for the BVP solver."""

import numpy as np
from numba import njit, prange

from .methods import _sor_kernel, _sor_redblack_kernel

_sor_kernel_jit = njit(cache=True)(_sor_kernel)


@njit(parallel=True, cache=True)
def _sor_redblack_kernel_parallel(y, is_sink, omega):
    """Parallel red-black SOR kernel.

    Each color sweep is embarrassingly parallel because all neighbors
    of a given color belong to the other color (not being updated).
    The outer j-loop uses prange for thread-level parallelism.
    """
    n_i, n_j = y.shape
    for color in range(2):
        for j in prange(1, n_j - 1):
            for i in range(n_i):
                if (i + j) % 2 != color:
                    continue
                if is_sink[i, j]:
                    continue
                i_plus = (i + 1) % n_i
                i_minus = (i - 1) % n_i
                y[i, j] = (
                    omega * 0.25 * (y[i_plus, j] + y[i_minus, j] + y[i, j + 1] + y[i, j - 1])
                    + (1 - omega) * y[i, j]
                )


def make_sor_numba_step(is_insulator, is_sink, omega: float, **kwargs):
    """Return a numba-accelerated SOR step function (sequential)."""
    if not (0 <= omega <= 2):
        raise ValueError(f"omega must be in [0, 2], got {omega:.2f}")

    if np.any(is_insulator):
        from .methods import make_sor_step
        return make_sor_step(is_insulator, is_sink, omega)

    def sor_numba_step(y, **kwargs):
        _sor_kernel_jit(y, is_sink, omega)
        return y

    return sor_numba_step


def make_sor_numba_redblack_step(is_insulator, is_sink, omega: float, **kwargs):
    """Return a numba-accelerated parallel red-black SOR step function."""
    if not (0 <= omega <= 2):
        raise ValueError(f"omega must be in [0, 2], got {omega:.2f}")

    if np.any(is_insulator):
        from .methods import make_sor_step
        return make_sor_step(is_insulator, is_sink, omega)

    def sor_numba_redblack_step(y, **kwargs):
        _sor_redblack_kernel_parallel(y, is_sink, omega)
        return y

    return sor_numba_redblack_step
