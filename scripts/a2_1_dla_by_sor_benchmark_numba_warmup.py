"""Benchmark: numba JIT warm-up cost for SOR-based DLA.

Compares sor_numba with and without prior JIT warm-up, using the best
configuration (gradient init + optimal omega). Shows how much of the
wall-clock time on a fresh run is JIT compilation vs actual work.
"""

import time
import subprocess
import sys
import numpy as np

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.models.dla_by_sor import grow_dla_sor
from scicomp3.bvp.omega import get_optimal_omega


def fixed_bc(k, y):
    apply_diffusion_bc(y)
    return y


def run_dla(method, N, n_steps, eta, seed, omega, tol, max_iter):
    """Run DLA with gradient init and return wall time, cluster size, iters."""
    grid = Grid2D(N=N, L=1.0)
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
OMEGA = get_optimal_omega(N)
TOL = 1e-4
MAX_ITER = 2_000

# -- Run in a fresh subprocess (no warm-up) -----------------------------------
# We spawn ourselves with a flag to measure numba cold-start cost accurately.
if "--subprocess" in sys.argv:
    t, c, iters = run_dla("sor_numba", N, N_STEPS, ETA, SEED, OMEGA, TOL, MAX_ITER)
    total = iters.sum()
    mean = iters.mean()
    mx = iters.max()
    at = int(iters.argmax()) + 1
    print(f"{t:.4f} {c} {total} {mean:.1f} {mx} {at}")
    sys.exit(0)

# -- Main: run all cases and display table ------------------------------------
print(f"Benchmark: N={N}, n_steps={N_STEPS}, eta={ETA}, omega={OMEGA:.4f}")
print(f"Config: sor_numba + gradient init + optimal omega\n")

header = (f"{'Case':<32} {'Time (s)':>10} {'Cluster':>10} {'Speedup':>10}"
          f" {'Total iter':>12} {'Mean/step':>10} {'Max/step':>10} {'@ step':>8}")
print(header)
print("-" * len(header))

results = {}

# 1. Pure Python SOR baseline (no numba involved)
t, c, iters = run_dla("sor", N, N_STEPS, ETA, SEED, OMEGA, TOL, MAX_ITER)
baseline_time = t
results["sor"] = (t, c, iters)
label = "sor (baseline)"
print(f"{label:<32} {t:>10.2f} {c:>10d} {'1.00x':>10}"
      f" {iters.sum():>12d} {iters.mean():>10.1f} {iters.max():>10d}"
      f" {int(iters.argmax()) + 1:>8d}")

# 2. numba WITHOUT warm-up (fresh subprocess, includes JIT compilation)
proc = subprocess.run(
    [sys.executable, __file__, "--subprocess"],
    capture_output=True, text=True,
)
parts = proc.stdout.strip().split()
t_cold = float(parts[0])
c_cold, total_cold, mean_cold_str, mx_cold, at_cold = (
    int(parts[1]), int(parts[2]), parts[3], int(parts[4]), int(parts[5])
)
mean_cold = float(mean_cold_str)
speedup = baseline_time / t_cold
label = "numba (no warm-up)"
print(f"{label:<32} {t_cold:>10.2f} {c_cold:>10d} {speedup:>9.2f}x"
      f" {total_cold:>12d} {mean_cold:>10.1f} {mx_cold:>10d} {at_cold:>8d}")

# 3. numba WITH warm-up (JIT already compiled in this process)
print("\nWarming up numba JIT...")
_ = run_dla("sor_numba", N, 2, ETA, SEED, OMEGA, TOL, MAX_ITER)
print("Warm-up done.\n")

# Re-print header for the warm case
print(header)
print("-" * len(header))

# Reprint baseline
t, c, iters = results["sor"]
label = "sor (baseline)"
print(f"{label:<32} {t:>10.2f} {c:>10d} {'1.00x':>10}"
      f" {iters.sum():>12d} {iters.mean():>10.1f} {iters.max():>10d}"
      f" {int(iters.argmax()) + 1:>8d}")

# Reprint cold
label = "numba (no warm-up)"
print(f"{label:<32} {t_cold:>10.2f} {c_cold:>10d} {baseline_time / t_cold:>9.2f}x"
      f" {total_cold:>12d} {mean_cold:>10.1f} {mx_cold:>10d} {at_cold:>8d}")

# Run warm
t_warm, c_warm, iters_warm = run_dla("sor_numba", N, N_STEPS, ETA, SEED, OMEGA, TOL, MAX_ITER)
speedup_warm = baseline_time / t_warm
label = "numba (warmed up)"
print(f"{label:<32} {t_warm:>10.2f} {c_warm:>10d} {speedup_warm:>9.2f}x"
      f" {iters_warm.sum():>12d} {iters_warm.mean():>10.1f} {iters_warm.max():>10d}"
      f" {int(iters_warm.argmax()) + 1:>8d}")

# Summary
print(f"\n--- Summary ---")
print(f"JIT warm-up overhead: {t_cold - t_warm:.2f}s ({t_cold / t_warm:.1f}x slower than warmed)")
print(f"numba warmed vs sor: {speedup_warm:.1f}x faster")
print(f"numba cold vs sor:   {baseline_time / t_cold:.1f}x faster (even with JIT cost)")
