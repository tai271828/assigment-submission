"""Accelerated SOR variants for BVP solvers.

Provides two faster alternatives to the scalar-loop SOR in methods.py:

- **Red-Black SOR** (``make_sor_redblack_step``):  Vectorised with NumPy.
  Updates even (i+j even) and odd (i+j odd) sites in two half-sweeps,
  each fully vectorised.  No Python element loop.

- **Numba SOR** (``make_sor_numba_step``):  JIT-compiled with Numba.
  Same sequential Gauss-Seidel ordering as the original ``make_sor_step``,
  but the inner double loop runs in compiled machine code.

Both follow the same factory-function convention used by the METHODS
registry: each takes ``(is_insulator, is_sink, omega, **kwargs)`` and
returns a ``step(y, **kwargs) -> y`` callable.

Note on Red-Black ordering:
    Standard (lexicographic) Gauss-Seidel SOR has a well-defined optimal ω.
    Red-Black SOR uses a different update ordering, which changes the
    iteration matrix.  For the 2D Laplace equation the *same* optimal ω
    formula applies (Young's theorem covers consistently-ordered splittings,
    and red-black is one such ordering).  Convergence rates are very similar
    in practice; the main advantage is vectorisability.
"""

import numpy as np


# ---------------------------------------------------------------------------
#  Red-Black SOR  (NumPy-vectorised)
# ---------------------------------------------------------------------------


def _build_checkerboard(shape):
    """Return boolean masks for red (i+j even) and black (i+j odd) sites."""
    ni, nj = shape
    ii, jj = np.ogrid[:ni, :nj]
    red = (ii + jj) % 2 == 0
    black = ~red
    return red, black


def make_sor_redblack_step(is_insulator, is_sink, omega: float, **kwargs):
    """Return a Red-Black SOR step function (fully vectorised).

    One step consists of two half-sweeps:
      1. Update all RED sites (i+j even) using current values at BLACK sites.
      2. Update all BLACK sites (i+j odd) using the just-updated RED sites.

    Each half-sweep is a single NumPy array operation — no Python loops.

    Boundary handling:
      - j=0 and j=N rows are never updated (Dirichlet BCs).
      - x-direction is periodic: ``np.roll(..., axis=0)`` wraps correctly.
      - Sink sites are reset to 0 after each half-sweep.
      - Insulator handling is supported but adds overhead.

    Args:
        is_insulator: Boolean mask (N+1, N+1).
        is_sink:      Boolean mask (N+1, N+1).
        omega:        SOR relaxation parameter in (0, 2).

    Returns:
        step(y, **kwargs) -> y   (modifies y in place).
    """
    if not (0 < omega < 2):
        raise ValueError(f"omega must be in (0, 2), got {omega:.4f}")

    shape = is_insulator.shape
    red_mask, black_mask = _build_checkerboard(shape)

    # Exclude boundaries (j=0, j=N), sinks, and insulators from updates
    interior = np.ones(shape, dtype=bool)
    interior[:, 0] = False
    interior[:, -1] = False

    red_update = red_mask & interior & ~is_sink & ~is_insulator
    black_update = black_mask & interior & ~is_sink & ~is_insulator

    has_insulator = np.any(is_insulator)

    if has_insulator:
        # Precompute per-point neighbour weights (exclude insulator neighbours)
        weights = _compute_neighbour_weights(is_insulator)

        def _neighbour_sum_masked(y):
            """Sum of non-insulator neighbours."""
            ins_val = y * is_insulator.astype(y.dtype)
            s = (
                np.roll(y, -1, axis=0)
                + np.roll(y, 1, axis=0)
                + np.roll(y, -1, axis=1)
                + np.roll(y, 1, axis=1)
            )
            s -= (
                np.roll(ins_val, -1, axis=0)
                + np.roll(ins_val, 1, axis=0)
                + np.roll(ins_val, -1, axis=1)
                + np.roll(ins_val, 1, axis=1)
            )
            return s

        def sor_redblack_step(y, **kwargs):
            # --- Red half-sweep ---
            s = _neighbour_sum_masked(y)
            gs_red = s / weights  # Gauss-Seidel value
            y[red_update] = omega * gs_red[red_update] + (1 - omega) * y[red_update]
            y[is_sink] = 0.0

            # --- Black half-sweep ---
            s = _neighbour_sum_masked(y)
            gs_black = s / weights
            y[black_update] = (
                omega * gs_black[black_update] + (1 - omega) * y[black_update]
            )
            y[is_sink] = 0.0
            return y

        return sor_redblack_step
    else:

        def sor_redblack_step(y, **kwargs):
            # --- Red half-sweep ---
            s = (
                np.roll(y, -1, axis=0)
                + np.roll(y, 1, axis=0)
                + np.roll(y, -1, axis=1)
                + np.roll(y, 1, axis=1)
            )
            gs_red = 0.25 * s
            y[red_update] = omega * gs_red[red_update] + (1 - omega) * y[red_update]
            y[is_sink] = 0.0

            # --- Black half-sweep ---
            s = (
                np.roll(y, -1, axis=0)
                + np.roll(y, 1, axis=0)
                + np.roll(y, -1, axis=1)
                + np.roll(y, 1, axis=1)
            )
            gs_black = 0.25 * s
            y[black_update] = (
                omega * gs_black[black_update] + (1 - omega) * y[black_update]
            )
            y[is_sink] = 0.0
            return y

        return sor_redblack_step


def _compute_neighbour_weights(is_insulator):
    """Count non-insulating neighbours per point (for weighted average)."""
    w = (
        np.roll(~is_insulator, -1, axis=0).astype(float)
        + np.roll(~is_insulator, 1, axis=0).astype(float)
        + np.roll(~is_insulator, -1, axis=1).astype(float)
        + np.roll(~is_insulator, 1, axis=1).astype(float)
    )
    w[is_insulator] = 1.0  # avoid /0; value unused
    return w


# ---------------------------------------------------------------------------
#  Numba SOR  (JIT-compiled scalar loop)
# ---------------------------------------------------------------------------


def _get_numba_kernels():
    """Lazy-import numba and compile the SOR kernels.

    Returns (sor_kernel, sor_kernel_with_sink) — both are @njit functions.
    Raises ImportError if numba is not installed.
    """
    import numba as nb

    @nb.njit(cache=True)
    def _sor_kernel(y, omega):
        """Plain SOR sweep — no sinks, no insulators."""
        ni, nj = y.shape
        for j in range(1, nj - 1):
            for i in range(ni):
                ip = (i + 1) % ni
                im = (i - 1) % ni
                y[i, j] = (
                    omega * 0.25 * (y[ip, j] + y[im, j] + y[i, j + 1] + y[i, j - 1])
                    + (1.0 - omega) * y[i, j]
                )
        return y

    @nb.njit(cache=True)
    def _sor_kernel_sink(y, omega, is_sink):
        """SOR sweep skipping sink sites."""
        ni, nj = y.shape
        for j in range(1, nj - 1):
            for i in range(ni):
                if is_sink[i, j]:
                    continue
                ip = (i + 1) % ni
                im = (i - 1) % ni
                y[i, j] = (
                    omega * 0.25 * (y[ip, j] + y[im, j] + y[i, j + 1] + y[i, j - 1])
                    + (1.0 - omega) * y[i, j]
                )
        return y

    @nb.njit(cache=True)
    def _sor_kernel_insulator(y, omega, is_insulator, is_sink):
        """SOR sweep with insulator and sink handling."""
        ni, nj = y.shape
        for j in range(1, nj - 1):
            for i in range(ni):
                if is_insulator[i, j] or is_sink[i, j]:
                    continue
                ip = (i + 1) % ni
                im = (i - 1) % ni
                total = 0.0
                count = 0
                for di, dj in [(ip, j), (im, j), (i, j + 1), (i, j - 1)]:
                    if not is_insulator[di, dj]:
                        total += y[di, dj]
                        count += 1
                if count > 0:
                    y[i, j] = omega * (total / count) + (1.0 - omega) * y[i, j]
        return y

    return _sor_kernel, _sor_kernel_sink, _sor_kernel_insulator


# Module-level cache so we compile only once
_numba_kernels = None


def _ensure_numba_kernels():
    global _numba_kernels
    if _numba_kernels is None:
        _numba_kernels = _get_numba_kernels()
    return _numba_kernels


def make_sor_numba_step(is_insulator, is_sink, omega: float, **kwargs):
    """Return a Numba-JIT SOR step function.

    Same sequential (lexicographic) ordering as ``make_sor_step`` but the
    inner double loop is JIT-compiled, giving ~50-100× speedup over the
    pure-Python version.

    First call triggers JIT compilation (~1-2 s); subsequent calls are fast.

    Args:
        is_insulator: Boolean mask (N+1, N+1).
        is_sink:      Boolean mask (N+1, N+1).
        omega:        SOR relaxation parameter in (0, 2).

    Returns:
        step(y, **kwargs) -> y   (modifies y in place).
    """
    if not (0 < omega < 2):
        raise ValueError(f"omega must be in (0, 2), got {omega:.4f}")

    kernel, kernel_sink, kernel_insulator = _ensure_numba_kernels()

    has_insulator = np.any(is_insulator)
    has_sink = np.any(is_sink)

    # Ensure masks are contiguous bool arrays for numba
    _is_insulator = np.ascontiguousarray(is_insulator)
    _is_sink = np.ascontiguousarray(is_sink)

    if has_insulator:

        def sor_numba_step(y, **kwargs):
            return kernel_insulator(y, omega, _is_insulator, _is_sink)

        return sor_numba_step
    elif has_sink:

        def sor_numba_step(y, **kwargs):
            return kernel_sink(y, omega, _is_sink)

        return sor_numba_step
    else:

        def sor_numba_step(y, **kwargs):
            return kernel(y, omega)

        return sor_numba_step


def warmup_numba(N=10):
    """Trigger JIT compilation on a tiny grid so first real call is fast."""
    kernels = _ensure_numba_kernels()
    y = np.zeros((N + 1, N + 1))
    sink = np.zeros((N + 1, N + 1), dtype=bool)
    ins = np.zeros((N + 1, N + 1), dtype=bool)
    kernels[0](y, 1.5)
    kernels[1](y, 1.5, sink)
    kernels[2](y, 1.5, ins, sink)
