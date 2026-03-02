"""Benchmark target for flamegraph profiling of DLA with different η values.

Stripped-down version of a2_1_dla_eta.py — no matplotlib, no saving images.
Runs two representative η configs (η=1 and η=3) with a smaller grid (N=50)
so that py-spy captures a statistically meaningful profile in under a minute.

Usage (from repo root):
    py-spy record --output flamegraph.svg -- \\
        .venv/bin/python scripts/bench_a2_1_dla_eta.py
"""

import numpy as np

from scicomp3.models.dla import run_dla
from scicomp3.bvp.omega import get_optimal_omega

N = 50
SEED = 7
OMEGA = get_optimal_omega(N)

# Two representative configs: standard DLA and high-η branched
BENCH_CONFIGS = [
    (1.0, 80, 800),   # (eta, n_steps, max_iter_per_step)
    (3.0, 60, 300),
]

for eta, n_steps, max_iter in BENCH_CONFIGS:
    print(f"eta={eta}  n_steps={n_steps}  max_iter={max_iter}", flush=True)
    rng = np.random.default_rng(SEED)
    cluster_mask, concentration, n_iters = run_dla(
        N=N, n_steps=n_steps, eta=eta, omega=OMEGA, max_iter=max_iter, rng=rng
    )
    n_cluster = cluster_mask.sum()
    mean_iters = float(np.mean(n_iters))
    print(f"  → {n_cluster} sites, mean SOR iters={mean_iters:.1f}", flush=True)

print("done")
