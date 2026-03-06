"""Benchmark: SOR variants for DLA growth."""

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

METHODS = [
    ("sor", "sor (python)"),
    ("sor_numba", "sor_numba"),
    ("sor_numba_redblack", "sor_numba_rb"),
]

# -- Warm up numba JIT -------------------------------------------------------
print("Warming up numba JIT compilation...")
_ = run_dla("sor_numba", N, 2, ETA, SEED, OMEGA, TOL, MAX_ITER)
_ = run_dla("sor_numba_redblack", N, 2, ETA, SEED, OMEGA, TOL, MAX_ITER)
print("Warm-up done.\n")

# -- Benchmark ---------------------------------------------------------------
print(f"Benchmark: N={N}, n_steps={N_STEPS}, eta={ETA}, omega={OMEGA:.4f}")
print(f"{'Method':<20} {'Time (s)':>10} {'Cluster':>10} {'Speedup':>10}")
print("-" * 55)

t_baseline = None
for method_key, label in METHODS:
    t, c = run_dla(method_key, N, N_STEPS, ETA, SEED, OMEGA, TOL, MAX_ITER)
    if t_baseline is None:
        t_baseline = t
        print(f"{label:<20} {t:>10.2f} {c:>10d} {'1.00x':>10}")
    else:
        speedup = t_baseline / t
        print(f"{label:<20} {t:>10.2f} {c:>10d} {speedup:>9.2f}x")
