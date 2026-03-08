"""Benchmark: pure-Python SOR vs numba-accelerated SOR for DLA growth.

Compares 8 cases across 3 independent variables:
  - Solver:  sor (Python loops) vs sor_numba (JIT-compiled)
  - Init:    c0 = zeros vs c0 = linear gradient (analytical empty-domain solution)
  - Omega:   optimal (theoretical) vs 1.0 (Gauss-Seidel, a worse guess)

Reports both wall-clock time and BVP iteration counts per growth step.
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


def run_dla(method, N, n_steps, eta, seed, omega, tol, max_iter, c0_mode="zeros"):
    """Run DLA and return wall time, cluster size, and per-step BVP iterations."""
    grid = Grid2D(N=N, L=1.0)
    growth_seed = (N // 2, N // 2)
    np.random.seed(seed)

    if c0_mode == "gradient":
        c0 = grid.Y / grid.L  # analytical solution c(y) = y for empty domain
    else:
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

    cluster_size = int(result.growth_mask.sum())
    return elapsed, cluster_size, result.bvp_iters_per_step


# -- Parameters --------------------------------------------------------------
N = 50
N_STEPS = 100
ETA = 1.0
SEED = 42
OMEGA_OPT = get_optimal_omega(N)
OMEGA_GS = 1.0  # Gauss-Seidel (worst reasonable omega for SOR)
TOL = 1e-4
MAX_ITER = 2_000

# -- Warm up numba JIT -------------------------------------------------------
print("Warming up numba JIT compilation...")
_ = run_dla("sor_numba", N, 2, ETA, SEED, OMEGA_OPT, TOL, MAX_ITER)
print("Warm-up done.\n")

# -- Benchmark ---------------------------------------------------------------
# (method, c0_mode, omega_value, label)
cases = [
    ("sor",       "zeros",    OMEGA_OPT, f"sor + zeros + w={OMEGA_OPT:.2f}"),
    ("sor",       "zeros",    OMEGA_GS,  f"sor + zeros + w={OMEGA_GS:.2f}"),
    ("sor",       "gradient", OMEGA_OPT, f"sor + grad + w={OMEGA_OPT:.2f}"),
    ("sor",       "gradient", OMEGA_GS,  f"sor + grad + w={OMEGA_GS:.2f}"),
    ("sor_numba", "zeros",    OMEGA_OPT, f"numba + zeros + w={OMEGA_OPT:.2f}"),
    ("sor_numba", "zeros",    OMEGA_GS,  f"numba + zeros + w={OMEGA_GS:.2f}"),
    ("sor_numba", "gradient", OMEGA_OPT, f"numba + grad + w={OMEGA_OPT:.2f}"),
    ("sor_numba", "gradient", OMEGA_GS,  f"numba + grad + w={OMEGA_GS:.2f}"),
]

print(f"Benchmark: N={N}, n_steps={N_STEPS}, eta={ETA}")
print(f"  omega_opt={OMEGA_OPT:.4f}, omega_gs={OMEGA_GS:.4f}")
print()
header = (f"{'Case':<32} {'Time (s)':>10} {'Cluster':>10} {'Speedup':>10}"
          f" {'Total iter':>12} {'Mean/step':>10} {'Max/step':>10} {'@ step':>8}")
print(header)
print("-" * len(header))

baseline_time = None
for method, c0_mode, omega, label in cases:
    t, c, iters = run_dla(method, N, N_STEPS, ETA, SEED, omega, TOL, MAX_ITER, c0_mode)
    if baseline_time is None:
        baseline_time = t
    speedup = baseline_time / t
    total_iter = iters.sum()
    mean_iter = iters.mean()
    max_iter_step = iters.max()
    max_at_step = int(iters.argmax()) + 1  # 1-indexed growth step
    print(f"{label:<32} {t:>10.2f} {c:>10d} {speedup:>9.2f}x"
          f" {total_iter:>12d} {mean_iter:>10.1f} {max_iter_step:>10d} {max_at_step:>8d}")
