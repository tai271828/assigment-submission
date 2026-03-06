"""Numba-accelerated SOR iteration for the BVP solver."""

import numpy as np
from numba import njit

from .methods import _sor_kernel

_sor_kernel_jit = njit(cache=True)(_sor_kernel)


def make_sor_numba_step(is_insulator, is_sink, omega: float, **kwargs):
    """Return a numba-accelerated SOR step function.

    Same semantics as make_sor_step but the inner loop is JIT-compiled.
    Only the no-insulator path is accelerated; if insulators are present,
    falls back to the pure-Python implementation.
    """
    if not (0 <= omega <= 2):
        raise ValueError(f"omega must be in [0, 2], got {omega:.2f}")

    if np.any(is_insulator):
        from .methods import make_sor_step
        return make_sor_step(is_insulator, is_sink, omega)

    def sor_numba_step(y, **kwargs):
        _sor_kernel_jit(y, is_sink, omega)
        return y

    return sor_numba_step
