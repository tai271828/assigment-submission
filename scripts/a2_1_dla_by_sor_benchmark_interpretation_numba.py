"""Benchmark interpretation: why does Numba speed up the SOR solver?

Isolates two loop patterns used in the SOR diffusion solver and benchmarks
them across four implementations:

  1. Pure Python (CPython interpreter)
  2. Python + Numba (JIT-compiled to native code)
  3. C via ctypes (includes marshalling overhead)
  4. C native executable (no marshalling overhead)

Loop patterns:
  - Single for-loop:  1D SOR-like relaxation along a row
  - Nested for-loop:  2D SOR-like relaxation over a grid (this is exactly
                      what commit 61dc5d4 accelerated with Numba)

The results show that Python's interpreter overhead dominates loop-heavy
code.  With a single loop (N iterations), the overhead is moderate.
With a nested loop (N^2 iterations), it compounds and becomes the
bottleneck.  Numba eliminates this overhead by JIT-compiling to native
machine code, achieving near-C performance.
"""

import ctypes
import subprocess
import time
from pathlib import Path

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# 1. Pure-Python kernels
# ---------------------------------------------------------------------------

def sor_1d_python(y, omega):
    """Single for-loop: 1D SOR relaxation."""
    n = len(y)
    for i in range(1, n - 1):
        y[i] = omega * 0.5 * (y[i - 1] + y[i + 1]) + (1 - omega) * y[i]


def sor_2d_python(y, omega):
    """Nested for-loop: 2D SOR relaxation (mirrors _sor_kernel in methods.py)."""
    nrows, ncols = y.shape
    for j in range(1, ncols - 1):
        for i in range(1, nrows - 1):
            y[i, j] = (
                omega * 0.25 * (y[i + 1, j] + y[i - 1, j] + y[i, j + 1] + y[i, j - 1])
                + (1 - omega) * y[i, j]
            )


# ---------------------------------------------------------------------------
# 2. Numba-accelerated kernels (same source, JIT-compiled)
# ---------------------------------------------------------------------------

@njit(cache=True)
def sor_1d_numba(y, omega):
    """Single for-loop: 1D SOR relaxation (Numba JIT)."""
    n = len(y)
    for i in range(1, n - 1):
        y[i] = omega * 0.5 * (y[i - 1] + y[i + 1]) + (1 - omega) * y[i]


@njit(cache=True)
def sor_2d_numba(y, omega):
    """Nested for-loop: 2D SOR relaxation (Numba JIT)."""
    nrows, ncols = y.shape
    for j in range(1, ncols - 1):
        for i in range(1, nrows - 1):
            y[i, j] = (
                omega * 0.25 * (y[i + 1, j] + y[i - 1, j] + y[i, j + 1] + y[i, j - 1])
                + (1 - omega) * y[i, j]
            )


# ---------------------------------------------------------------------------
# 3. C kernels (loaded via ctypes)
# ---------------------------------------------------------------------------

_LIB_PATH = Path(__file__).resolve().parent.parent / "contrib" / "sor_kernels.so"
_lib = ctypes.CDLL(str(_LIB_PATH))

# void sor_1d_kernel(double *y, int n, double omega)
_lib.sor_1d_kernel.restype = None
_lib.sor_1d_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double,
]

# void sor_2d_kernel(double *y, int nrows, int ncols, double omega)
_lib.sor_2d_kernel.restype = None
_lib.sor_2d_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
]


def sor_1d_c(y, omega):
    """Single for-loop: 1D SOR via C."""
    _lib.sor_1d_kernel(
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(y),
        omega,
    )


def sor_2d_c(y, omega):
    """Nested for-loop: 2D SOR via C."""
    nrows, ncols = y.shape
    _lib.sor_2d_kernel(
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        nrows,
        ncols,
        omega,
    )


# ---------------------------------------------------------------------------
# 4. C native executable (no ctypes marshalling overhead)
# ---------------------------------------------------------------------------

_BENCH_EXE = Path(__file__).resolve().parent.parent / "contrib" / "sor_benchmark"


def run_c_native(n_1d, repeats_1d, n_2d, repeats_2d, omega):
    """Run the native C benchmark executable and return (t_1d, t_2d)."""
    result = subprocess.run(
        [str(_BENCH_EXE), str(n_1d), str(repeats_1d), str(n_2d), str(repeats_2d), str(omega)],
        capture_output=True, text=True, check=True,
    )
    times = {}
    for line in result.stdout.strip().splitlines():
        key, val = line.split()
        times[key] = float(val)
    return times["1d"], times["2d"]


# ---------------------------------------------------------------------------
# Benchmarking harness
# ---------------------------------------------------------------------------

def benchmark(func, make_data, n_repeats, warmup=0):
    """Time a kernel over n_repeats calls, returning the total elapsed time."""
    # Warmup runs (not timed)
    for _ in range(warmup):
        data = make_data()
        func(*data)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        data = make_data()
        func(*data)
    return time.perf_counter() - t0


def print_table(label, results, n_repeats):
    """Print a comparison table for one benchmark."""
    print(f"\n{'=' * 62}")
    print(f"  {label}  ({n_repeats} iterations)")
    print(f"{'=' * 62}")
    print(f"  {'Implementation':<22} {'Total (s)':>10} {'Per-call (us)':>14} {'Speedup':>10}")
    print(f"  {'-' * 58}")
    t_python = results[0][1]
    for name, elapsed in results:
        per_call_us = elapsed / n_repeats * 1e6
        speedup = t_python / elapsed
        print(f"  {name:<22} {elapsed:>10.4f} {per_call_us:>14.1f} {speedup:>9.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OMEGA = 1.85
    N_1D = 1000       # length of 1D array
    N_2D = 100        # grid side length for 2D (100x100)
    REPEATS_1D = 1000
    REPEATS_2D = 100

    # -- Warm up Numba JIT ---------------------------------------------------
    print("Warming up Numba JIT compilation...")
    _y1 = np.linspace(0, 1, 10)
    sor_1d_numba(_y1, OMEGA)
    _y2 = np.linspace(0, 1, 100).reshape(10, 10)
    sor_2d_numba(_y2, OMEGA)
    print("Warm-up done.\n")

    # -- Run native C benchmark (no ctypes overhead) -------------------------
    print("Running native C benchmark executable...")
    t_c_native_1d, t_c_native_2d = run_c_native(
        N_1D, REPEATS_1D, N_2D, REPEATS_2D, OMEGA
    )
    print("Native C benchmark done.\n")

    # -- Benchmark 1: Single for-loop (1D) ----------------------------------
    def make_1d():
        return (np.linspace(0, 1, N_1D), OMEGA)

    results_1d = [
        ("Python (CPython)",     benchmark(sor_1d_python, make_1d, REPEATS_1D)),
        ("Python + Numba (JIT)", benchmark(sor_1d_numba,  make_1d, REPEATS_1D)),
        ("C via ctypes",         benchmark(sor_1d_c,      make_1d, REPEATS_1D)),
        ("C native (gcc -O2)",   t_c_native_1d),
    ]
    print_table(f"Single for-loop — 1D SOR (N={N_1D})", results_1d, REPEATS_1D)

    # -- Benchmark 2: Nested for-loop (2D) ----------------------------------
    def make_2d():
        y = np.zeros((N_2D + 1, N_2D + 1))
        y[:, -1] = 1.0  # top BC
        return (y, OMEGA)

    results_2d = [
        ("Python (CPython)",     benchmark(sor_2d_python, make_2d, REPEATS_2D)),
        ("Python + Numba (JIT)", benchmark(sor_2d_numba,  make_2d, REPEATS_2D)),
        ("C via ctypes",         benchmark(sor_2d_c,      make_2d, REPEATS_2D)),
        ("C native (gcc -O2)",   t_c_native_2d),
    ]
    print_table(f"Nested for-loop — 2D SOR ({N_2D}x{N_2D})", results_2d, REPEATS_2D)

    # -- Interpretation summary ----------------------------------------------
    t_py_1d  = results_1d[0][1]
    t_nb_1d  = results_1d[1][1]
    t_cn_1d  = results_1d[3][1]
    t_py_2d  = results_2d[0][1]
    t_nb_2d  = results_2d[1][1]
    t_cn_2d  = results_2d[3][1]

    print(f"\n{'=' * 70}")
    print("  Interpretation (using C native — no ctypes overhead)")
    print(f"{'=' * 70}")
    print(f"  1D loop (N={N_1D}):  Python/C = {t_py_1d/t_cn_1d:.0f}x,  "
          f"Numba/C = {t_nb_1d/t_cn_1d:.1f}x")
    print(f"  2D loop ({N_2D}x{N_2D}): Python/C = {t_py_2d/t_cn_2d:.0f}x,  "
          f"Numba/C = {t_nb_2d/t_cn_2d:.1f}x")
    print()
    print("  The nested loop amplifies Python's per-iteration interpreter")
    print("  overhead (type checks, object boxing, dict lookups) by N^2.")
    print("  Numba eliminates this by compiling to LLVM IR -> native code,")
    print("  achieving near-C performance with zero source code changes.")
