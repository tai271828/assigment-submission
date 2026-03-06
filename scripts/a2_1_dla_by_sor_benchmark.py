"""Benchmark: pure-Python SOR vs numba-accelerated SOR for DLA growth."""

import time
import numpy as np

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.models.dla_by_sor import grow_dla_sor
from scicomp3.bvp.omega import get_optimal_omega


def fixed_bc(k, y):
    apply_diffusion_bc(y)
    return y


def run_dla(method, N, n_steps, eta, seed, omega, tol, max_iter):
    grid = Grid2D(N=N, L=1.0)
    growth_seed = (N // 2, N // 2)
    np.random.seed(seed)
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)

    t0 = time.perf_counter()
    result = grow_dla_sor(
        n_steps,
        growth_seed,
        eta,
        c0,
        omega,
        tol,
        max_iter_sor=max_iter,
        post_step=fixed_bc,
        method=method,
    )
    elapsed = time.perf_counter() - t0
    cluster_size = result.growth_mask.sum()
    return elapsed, cluster_size


# -- Parameters --------------------------------------------------------------
N = 50
N_STEPS = 100
ETA = 1.0
SEED = 42
OMEGA = get_optimal_omega(N)
TOL = 1e-4
MAX_ITER = 2_000

# -- Warm up numba JIT -------------------------------------------------------
print("Warming up numba JIT compilation...")
_ = run_dla("sor_numba", N, 2, ETA, SEED, OMEGA, TOL, MAX_ITER)
print("Warm-up done.\n")

# -- Benchmark ---------------------------------------------------------------
print(f"Benchmark: N={N}, n_steps={N_STEPS}, eta={ETA}, omega={OMEGA:.4f}")
print(f"{'Method':<15} {'Time (s)':>10} {'Cluster':>10} {'Speedup':>10}")
print("-" * 50)

t_python, c_python = run_dla("sor", N, N_STEPS, ETA, SEED, OMEGA, TOL, MAX_ITER)
print(f"{'sor (python)':<15} {t_python:>10.2f} {c_python:>10d} {'1.00x':>10}")

t_numba, c_numba = run_dla("sor_numba", N, N_STEPS, ETA, SEED, OMEGA, TOL, MAX_ITER)
speedup = t_python / t_numba
print(f"{'sor_numba':<15} {t_numba:>10.2f} {c_numba:>10d} {speedup:>9.2f}x")
