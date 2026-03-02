"""Profiling target for flamegraph: original SOR DLA (no numba/red-black).

Run with py-spy:
    py-spy record -o images/figures/flamegraph_sor_original.svg \
        --subprocesses -- uv run python scripts/a2_1_dla_flamegraph.py
"""

import numpy as np
from scicomp3.models.dla import run_dla
from scicomp3.bvp.omega import get_optimal_omega

N = 100
N_STEPS = 50
OMEGA = get_optimal_omega(N)
rng = np.random.default_rng(42)

print(f"Profiling original SOR: N={N}, steps={N_STEPS}, omega={OMEGA:.4f}")

cluster_mask, concentration, n_iters = run_dla(
    N=N,
    n_steps=N_STEPS,
    eta=1.0,
    omega=OMEGA,
    rng=rng,
    method="sor",  # original pure-Python SOR
)

print(f"Done. Cluster={cluster_mask.sum()} sites, "
      f"mean iters/step={np.mean(n_iters):.0f}")
