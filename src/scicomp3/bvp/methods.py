"""Iterative methods for BVP (steady-state) solvers.

Each method is provided as a setup function that takes the grid configuration
(insulator mask, sink_mask, omega, etc.) and returns the iteration step function to be
used by the solver. All returned step functions have a uniform signature and
can be used interchangeably.

Setup functions:
- make_jacobi_step:       Jacobi iteration (vectorised; returns a new array)
- make_gauss_seidel_step: Gauss-Seidel iteration (updates in place)
- make_sor_step:          Successive Over-Relaxation (updates in place;
                          requires omega in [0, 2])

All returned step functions have signature:
    step(y, **kwargs) -> y

If the grid contains insulating objects, the setup function returns a variant
that excludes insulating neighbours from the average. Otherwise, a faster
variant without insulator handling is returned.

The METHODS registry maps string keys to setup functions
"""

import numpy as np

def _compute_jacobi_weights(is_insulator):
    """
    For each point, count non-insulating neighbours.
    Returns a float array; insulator points get weight 1 to avoid division.
    """
    neighbour_count = (
        np.roll(~is_insulator, -1, axis=0).astype(float) +
        np.roll(~is_insulator,  1, axis=0).astype(float) +
        np.roll(~is_insulator, -1, axis=1).astype(float) +
        np.roll(~is_insulator,  1, axis=1).astype(float)
    )
    # Avoid division by zero at insulator points (value won't be used)
    neighbour_count[is_insulator] = 1.0
    return neighbour_count


def make_jacobi_step(is_insulator, is_sink, **kwargs):
    """Return a Jacobi iteration step function.

    The returned function computes one Jacobi iteration step (Eq. 12):

        c^{k+1}_{i,j} = (1/4)(c^k_{i+1,j} + c^k_{i-1,j} + c^k_{i,j+1} + c^k_{i,j-1})

    Uses np.roll for the five-point stencil (periodic in x via wrap-around).
    Requires two arrays — the returned function does not modify y in place.

    If insulators are present, insulating neighbours are excluded from the
    average and the denominator is adjusted accordingly. Insulator points
    are left unchanged.

    Args:
        is_insulator: Boolean array of shape (N+1 x N+1), True at insulating
                      grid points.
        is_sink: Boolean array of shape (N+1 x N+1), True at sink grid points.

    Returns:
        step: A function with signature step(y, **kwargs) -> y_new that
              performs one Jacobi iteration step.
    """
    if np.any(is_insulator):
        jacobi_weights = _compute_jacobi_weights(is_insulator)

        def jacobi_step_with_insulator(y, **kwargs):
            neighbour_sum = (
                np.roll(y, -1, axis=0) + np.roll(y, 1, axis=0) +
                np.roll(y, -1, axis=1) + np.roll(y, 1, axis=1)
            )
            # Zero out insulating neighbours' contributions
            insulator_zeroed = y * is_insulator.astype(float)
            neighbour_sum -= (
                np.roll(insulator_zeroed, -1, axis=0) + np.roll(insulator_zeroed,  1, axis=0) +
                np.roll(insulator_zeroed, -1, axis=1) + np.roll(insulator_zeroed,  1, axis=1)
            )
            y_new = neighbour_sum / jacobi_weights
            y_new[is_insulator] = y[is_insulator]   # leave insulator points unchanged
            y_new[is_sink] = 0                      # And keep sinks at zero
            return y_new
        return jacobi_step_with_insulator
    else:
        def jacobi_step(y, **kwargs):
            y_new = 0.25 * (
                np.roll(y, -1, axis=0) + np.roll(y, 1, axis=0) +
                np.roll(y, -1, axis=1) + np.roll(y, 1, axis=1)
            )
            # Keep sinks at zero
            y_new[is_sink] = 0
            return y_new
        return jacobi_step


def make_gauss_seidel_step(is_insulator, is_sink, **kwargs):
    """Return a Gauss-Seidel iteration step function.

    The returned function computes one Gauss-Seidel iteration step (Sec. 1.5):

        c^{k+1}_{i,j} = (1/4)(c^k_{i+1,j} + c^{k+1}_{i-1,j}
                             + c^k_{i,j+1} + c^{k+1}_{i,j-1})

    Uses already-updated values as soon as they are available.
    Sweeps rows: incrementing i for fixed j (as stated in the assignment).
    The returned function updates y in place and returns the same array.

    Periodic x boundary is handled via modular indexing.

    If insulators are present, insulating neighbours are excluded from the
    average and the denominator is adjusted accordingly. Insulator points
    are left unchanged.

    Args:
        is_insulator: Boolean array of shape (N+1 x N+1), True at insulating
                      grid points.
        is_sink: Boolean array of shape (N+1 x N+1), True at sink grid points.

    Returns:
        step: A function with signature step(y, **kwargs) -> y that performs
              one Gauss-Seidel iteration step, modifying y in place.
    """
    if np.any(is_insulator):
        def gauss_seidel_step_with_insulator(y, **kwargs):
            n_i, n_j = y.shape
            for j in range(1, n_j - 1):        # interior y-points
                for i in range(n_i):            # all x-points (periodic)
                    if is_insulator[i, j] or is_sink[i, j]:
                        continue
                    i_plus = (i + 1) % n_i
                    i_minus = (i - 1) % n_i
                    coords = [(i_plus, j), (i_minus, j), (i, j + 1), (i, j - 1)]
                    new_value = np.mean([y[coord] for coord in coords if not is_insulator[coord]])
                    if not np.isnan(new_value):
                        y[i,j] = new_value
            return y
        return gauss_seidel_step_with_insulator
    else:
        def gauss_seidel_step(y, **kwargs):
            n_i, n_j = y.shape
            for j in range(1, n_j - 1):        # interior y-points
                for i in range(n_i):            # all x-points (periodic)
                    if is_sink[i, j]:
                        continue
                    i_plus = (i + 1) % n_i
                    i_minus = (i - 1) % n_i
                    y[i, j] = 0.25 * (y[i_plus, j] + y[i_minus, j] +
                                    y[i, j + 1] + y[i, j - 1])
            return y
        return gauss_seidel_step


def _sor_kernel(y, is_sink, omega):
    """SOR inner loop (no insulator case).

    Sweeps rows: incrementing i for fixed j, with periodic x boundary.
    Updates y in place.
    """
    n_i, n_j = y.shape
    for j in range(1, n_j - 1):
        for i in range(n_i):
            if is_sink[i, j]:
                continue
            i_plus = (i + 1) % n_i
            i_minus = (i - 1) % n_i
            y[i, j] = (
                omega * 0.25 * (y[i_plus, j] + y[i_minus, j] + y[i, j + 1] + y[i, j - 1])
                + (1 - omega) * y[i, j]
            )


def _sor_redblack_kernel(y, is_sink, omega):
    """Red-black SOR inner loop (no insulator case).

    Splits the grid into two independent sets using checkerboard coloring:
    red points ((i+j) even) and black points ((i+j) odd). Each color's
    sweep only reads from the other color, so within a sweep all updates
    are independent and parallelizable.

    Updates y in place.
    """
    n_i, n_j = y.shape
    for color in range(2):
        for j in range(1, n_j - 1):
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


def make_sor_redblack_step(is_insulator, is_sink, omega: float, **kwargs):
    """Return a red-black SOR step function.

    Same convergence properties as standard SOR, but the update order
    (checkerboard coloring) makes each half-sweep embarrassingly parallel.
    Falls back to pure-Python standard SOR if insulators are present.
    """
    if not (0 <= omega <= 2):
        raise ValueError(f"omega must be in [0, 2], got {omega:.2f}")

    if np.any(is_insulator):
        return make_sor_step(is_insulator, is_sink, omega)

    def sor_redblack_step(y, **kwargs):
        _sor_redblack_kernel(y, is_sink, omega)
        return y

    return sor_redblack_step


def make_sor_step(is_insulator, is_sink, omega: float, **kwargs):
    """Return a Successive Over-Relaxation (SOR) iteration step function.

    The returned function computes one SOR step:

        c^{k+1}_{i,j} = (ω/4)(c^k_{i+1,j} + c^{k+1}_{i-1,j}
                              + c^k_{i,j+1} + c^{k+1}_{i,j-1})
                        + (1-ω) c^k_{i,j}

    SOR generalises Gauss-Seidel: setting omega=1 recovers Gauss-Seidel exactly.
    Values of omega in (1, 2) over-relax and typically accelerate convergence;
    values in (0, 1) under-relax. omega must be in [0, 2].

    Uses already-updated values as soon as they are available.
    Sweeps rows: incrementing i for fixed j (as stated in the assignment).
    The returned function updates y in place and returns the same array.

    Periodic x boundary is handled via modular indexing.

    If insulators are present, insulating neighbours are excluded from the
    average and the denominator is adjusted accordingly. Insulator points
    are left unchanged.

    Args:
        is_insulator: Boolean array of shape (N+1 x N+1), True at insulating
                      grid points.
        is_sink: Boolean array of shape (N+1 x N+1), True at sink grid points.
        omega: Relaxation parameter, must be in [0, 2].

    Raises:
        ValueError: If omega is outside [0, 2].

    Returns:
        step: A function with signature step(y, **kwargs) -> y that performs
              one SOR iteration step, modifying y in place.
    """
    if not (0 <= omega <= 2):
        raise ValueError(f"omega must be in [0, 2], got {omega:.2f}")

    # Check if there are any insulating objects
    if np.any(is_insulator):
        def sor_step_with_insulator(y, **kwargs):
            n_i, n_j = y.shape
            for j in range(1, n_j - 1):        # interior y-points
                for i in range(n_i):            # all x-points (periodic)
                    if is_insulator[i, j] or is_sink[i, j]:
                        continue
                    i_plus = (i + 1) % n_i
                    i_minus = (i - 1) % n_i
                    coords = [(i_plus, j), (i_minus, j), (i, j + 1), (i, j - 1)]
                    new_value = np.mean([y[coord] for coord in coords if not is_insulator[coord]])
                    if not np.isnan(new_value):
                        y[i,j] = omega * new_value + (1 - omega) * y[i, j]
            return y
        return sor_step_with_insulator
    else:
        def sor_step(y, **kwargs):
            _sor_kernel(y, is_sink, omega)
            return y
        return sor_step

METHODS = {
    "jacobi": make_jacobi_step,
    "gauss_seidel": make_gauss_seidel_step,
    "sor": make_sor_step,
    "sor_redblack": make_sor_redblack_step,
}

try:
    from .methods_numba import make_sor_numba_step, make_sor_numba_redblack_step
    METHODS["sor_numba"] = make_sor_numba_step
    METHODS["sor_numba_redblack"] = make_sor_numba_redblack_step
except ImportError:
    pass
