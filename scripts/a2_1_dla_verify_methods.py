"""Verify consistency of all three SOR variants for the BVP solver.

Runs the same DLA problem with:
  1. Original SOR (Python loop, lexicographic ordering)
  2. Red-Black SOR (NumPy vectorised, checkerboard ordering)
  3. Numba SOR (JIT-compiled, lexicographic ordering)

Checks:
  - All three produce the same cluster (identical RNG → identical growth).
  - Concentration fields agree within tolerance.
  - Reports iteration counts and wall times for comparison.

Note: Red-Black uses different update ordering than lexicographic SOR, so
the converged concentration fields may differ at machine precision level.
However, with the same RNG seed and tolerance, the growth sequence should
be identical because the converged fields are close enough that the
discrete growth-probability sampling produces the same choices.
"""

import time
import numpy as np

from scicomp3.models.dla import run_dla
from scicomp3.bvp.omega import get_optimal_omega
from scicomp3.bvp.methods_fast import warmup_numba

# ── Parameters ──────────────────────────────────────────────────────────────
N = 50  # small grid for fast testing
N_STEPS = 100
ETA = 1.0
SEED = 42
TOL = 1e-5  # tight tolerance so all methods agree closely
OMEGA = get_optimal_omega(N)

print("=" * 70)
print("SOR Variant Consistency Verification")
print("=" * 70)
print(f"Grid: {N}×{N},  steps: {N_STEPS},  η={ETA},  ω={OMEGA:.4f},  tol={TOL}")
print()

# ── Warm up Numba ────────────────────────────────────────────────────────────
print("Warming up Numba JIT...")
warmup_numba()
print()

# ── Run all three methods ────────────────────────────────────────────────────
methods = {
    "sor": "Original SOR (Python loop)",
    "sor_redblack": "Red-Black SOR (NumPy vectorised)",
    "sor_numba": "Numba SOR (JIT-compiled)",
}

results = {}
for method_key, method_name in methods.items():
    rng = np.random.default_rng(SEED)  # fresh RNG each time for reproducibility
    print(f"Running {method_name}...")
    t0 = time.perf_counter()
    cluster_mask, concentration, n_iters = run_dla(
        N=N,
        n_steps=N_STEPS,
        eta=ETA,
        omega=OMEGA,
        tol=TOL,
        rng=rng,
        method=method_key,
    )
    elapsed = time.perf_counter() - t0
    results[method_key] = {
        "cluster": cluster_mask,
        "concentration": concentration,
        "n_iters": n_iters,
        "time": elapsed,
        "name": method_name,
    }
    n_cluster = cluster_mask.sum()
    print(
        f"  Cluster size: {n_cluster},  "
        f"mean iters/step: {np.mean(n_iters):.1f},  "
        f"wall time: {elapsed:.2f} s"
    )
    print()

# ── Compare results ──────────────────────────────────────────────────────────
print("=" * 70)
print("COMPARISON")
print("=" * 70)

ref_key = "sor"
ref = results[ref_key]

all_pass = True

for method_key in ["sor_redblack", "sor_numba"]:
    r = results[method_key]
    print(f"\n{r['name']} vs {ref['name']}:")

    # 1. Cluster identity
    cluster_match = np.array_equal(ref["cluster"], r["cluster"])
    n_diff_sites = np.sum(ref["cluster"] != r["cluster"])
    status = "PASS ✓" if cluster_match else f"FAIL ✗ ({n_diff_sites} sites differ)"
    print(f"  Cluster identical:    {status}")
    if not cluster_match:
        all_pass = False

    # 2. Concentration field
    max_diff = np.max(np.abs(ref["concentration"] - r["concentration"]))
    conc_close = max_diff < 1e-3  # generous threshold (different ordering)
    status = "PASS ✓" if conc_close else "FAIL ✗"
    print(f"  Max |Δc|:             {max_diff:.2e}  {status}")
    if not conc_close:
        all_pass = False

    # 3. Iteration counts (may differ for red-black due to different ordering)
    iter_ref = np.array(ref["n_iters"])
    iter_cmp = np.array(r["n_iters"])
    if len(iter_ref) == len(iter_cmp):
        max_iter_diff = np.max(np.abs(iter_ref - iter_cmp))
        mean_iter_diff = np.mean(np.abs(iter_ref - iter_cmp))
        print(
            f"  Iteration counts:     max diff = {max_iter_diff}, "
            f"mean diff = {mean_iter_diff:.1f}"
        )
    else:
        print(
            f"  Iteration counts:     different lengths "
            f"({len(iter_ref)} vs {len(iter_cmp)})"
        )

    # 4. Speedup
    speedup = ref["time"] / r["time"] if r["time"] > 0 else float("inf")
    print(
        f"  Speedup:              {speedup:.1f}×  "
        f"({ref['time']:.2f}s → {r['time']:.2f}s)"
    )

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"OVERALL: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
print("=" * 70)

# Time comparison table
print()
print(
    f"{'Method':<35} {'Time (s)':>10} {'Speedup':>10} {'Cluster':>10} {'Mean iter':>10}"
)
print("-" * 75)
for method_key, r in results.items():
    speedup = ref["time"] / r["time"] if r["time"] > 0 else 0
    cluster_ok = "✓" if np.array_equal(ref["cluster"], r["cluster"]) else "✗"
    print(
        f"{r['name']:<35} {r['time']:>10.2f} {speedup:>9.1f}× {cluster_ok:>10} "
        f"{np.mean(r['n_iters']):>10.1f}"
    )
