"""Benchmark: numba SOR-based DLA scaling with grid size N.

Uses the best configuration (numba + gradient init + optimal omega).
Sweeps N from 100 to 1000 and reports wall time, iteration counts,
and the top 3 most expensive growth steps per run.
"""

import time
import numpy as np

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.models.dla_by_sor import grow_dla_sor
from scicomp3.bvp.omega import get_optimal_omega


def fixed_bc(k, y):
    apply_diffusion_bc(y)
    return y


def run_dla(N, n_steps, eta, seed, tol, max_iter):
    """Run numba DLA with gradient init and optimal omega."""
    grid = Grid2D(N=N, L=1.0)
    omega = get_optimal_omega(N)
    growth_seed = (N // 2, N // 2)
    np.random.seed(seed)

    c0 = grid.Y / grid.L
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
        method="sor_numba",
    )
    elapsed = time.perf_counter() - t0

    cluster_size = int(result.growth_mask.sum())
    return elapsed, cluster_size, result.bvp_iters_per_step, omega


# -- Parameters --------------------------------------------------------------
N_VALUES = [100, 200, 300, 500, 750, 1000]
N_STEPS = 100
ETA = 1.0
SEED = 42
TOL = 1e-4
MAX_ITER = 10_000

# -- Warm up numba JIT -------------------------------------------------------
print("Warming up numba JIT compilation...")
_ = run_dla(50, 2, ETA, SEED, TOL, MAX_ITER)
print("Warm-up done.\n")

# -- Benchmark ---------------------------------------------------------------
print(f"Benchmark: n_steps={N_STEPS}, eta={ETA}, method=sor_numba")
print(f"Config: gradient init + optimal omega per N\n")

header = (f"{'N':>6} {'omega':>8} {'Time (s)':>10} {'Cluster':>10}"
          f" {'Total iter':>12} {'Mean/step':>10} {'Max/step':>10}"
          f" {'Top-3 steps (step: iters)':>40}")
print(header)
print("-" * len(header))

for N in N_VALUES:
    t, c, iters, omega = run_dla(N, N_STEPS, ETA, SEED, TOL, MAX_ITER)

    total_iter = iters.sum()
    mean_iter = iters.mean()
    max_iter_val = iters.max()

    # Top 3 most expensive steps (1-indexed)
    top3_idx = np.argsort(iters)[-3:][::-1]
    top3_str = ", ".join(f"{idx + 1}: {iters[idx]}" for idx in top3_idx)

    print(f"{N:>6d} {omega:>8.4f} {t:>10.2f} {c:>10d}"
          f" {total_iter:>12d} {mean_iter:>10.1f} {max_iter_val:>10d}"
          f"   {top3_str}")
