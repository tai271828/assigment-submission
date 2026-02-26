"""DLA solver performance: SOR vs Jacobi, warm-start benefit (Assignment 2.1.B).

Assignment question B asks to reduce the time for solving the diffusion equation
in the DLA loop and test larger grids.

Key insight: the existing SOR uses Python loops → O(N²) Python iterations per
BVP step.  The Jacobi method (fully vectorised via numpy) runs O(1) numpy calls
per step, which is far faster per iteration even though Jacobi needs more
iterations to converge than SOR.

This script benchmarks:
  1. SOR warm-start vs cold-start for N=100 (shows warm-start benefit)
  2. Jacobi (vectorised) vs SOR for N=100 DLA (shows relative speed)
  3. Scaling: SOR wall-time per DLA step for N ∈ {50, 75, 100}
     (shows why larger grids need a faster solver)
"""

import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots  # noqa: F401

styles = ["science"] if (shutil.which("latex") and shutil.which("dvipng")) else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.bvp.solver import solve_bvp
from scicomp3.bvp.omega import get_optimal_omega
from scicomp3.models.dla import run_dla, find_growth_candidates, _bvp_bc

SEED = 42

# ── Helpers ──────────────────────────────────────────────────────────────────

def _bvp_bc_wrapper(k, c):
    return _bvp_bc(k, c)


def time_bvp_solve(y0, method, sink_coords, omega, tol=1e-4, max_iter=2000):
    """Time one BVP solve; return (result, elapsed_s)."""
    t0 = time.perf_counter()
    res = solve_bvp(
        y0, method=method, tol=tol, max_iter=max_iter,
        post_step=_bvp_bc_wrapper,
        sink_coordinates=sink_coords,
        **({} if method == "jacobi" else {"omega": omega}),
    )
    return res, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Warm-start vs cold-start for SOR on N=100 after 50 DLA steps
# ═══════════════════════════════════════════════════════════════════════════════
print("=== 1. Warm-start vs cold-start (SOR, N=100, 50 steps) ===")
N = 100
OMEGA = get_optimal_omega(N)
rng   = np.random.default_rng(SEED)

# Build a cluster of 50 sites
cluster_mask_50, warm_conc, _ = run_dla(
    N=N, n_steps=50, omega=OMEGA, rng=rng
)
sink_coords = np.argwhere(cluster_mask_50)

# Linear-gradient initial guess (cold start)
y_coords  = np.linspace(0, 1, N + 1)
cold_conc = np.broadcast_to(y_coords, (N + 1, N + 1)).copy()

_, t_cold = time_bvp_solve(cold_conc,   "sor", sink_coords, OMEGA)
_, t_warm = time_bvp_solve(warm_conc,   "sor", sink_coords, OMEGA)
r_cold    = solve_bvp(cold_conc, method="sor", tol=1e-4, max_iter=2000,
                      post_step=_bvp_bc_wrapper, sink_coordinates=sink_coords,
                      omega=OMEGA)
r_warm    = solve_bvp(warm_conc, method="sor", tol=1e-4, max_iter=2000,
                      post_step=_bvp_bc_wrapper, sink_coordinates=sink_coords,
                      omega=OMEGA)

print(f"  Cold start: {r_cold.n_iter} iter,  {t_cold:.3f} s")
print(f"  Warm start: {r_warm.n_iter} iter,  {t_warm:.3f} s")
print(f"  Speed-up: {t_cold/t_warm:.1f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SOR (warm) vs Jacobi (vectorised) for DLA, N=100, 30 steps each
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. SOR (warm) vs Jacobi (vectorised) for DLA, N=100, 30 steps ===")
N_STEPS_BENCH = 30

t0 = time.perf_counter()
_, _, sor_iters = run_dla(N=N, n_steps=N_STEPS_BENCH, omega=OMEGA,
                           rng=np.random.default_rng(SEED))
t_sor = time.perf_counter() - t0
print(f"  SOR  warm: {t_sor:.2f} s   mean iters/step={np.mean(sor_iters):.0f}")


def run_dla_jacobi(N, n_steps, tol=1e-4, max_iter=5000, rng=None):
    """Variant of run_dla that uses the vectorised Jacobi method."""
    from scicomp3.models.dla import (
        find_growth_candidates, compute_growth_probabilities
    )
    if rng is None:
        rng = np.random.default_rng()
    cluster_mask = np.zeros((N + 1, N + 1), dtype=bool)
    cluster_mask[N // 2, 1] = True
    y_coords      = np.linspace(0.0, 1.0, N + 1)
    concentration = np.broadcast_to(y_coords, (N + 1, N + 1)).copy()
    n_iters = []
    for _ in range(n_steps):
        sink_coords = np.argwhere(cluster_mask)
        result = solve_bvp(
            concentration, method="jacobi", tol=tol, max_iter=max_iter,
            post_step=_bvp_bc_wrapper, sink_coordinates=sink_coords,
        )
        concentration = result.y
        n_iters.append(result.n_iter)
        candidates = find_growth_candidates(cluster_mask)
        if not np.any(candidates):
            break
        candidate_indices = np.argwhere(candidates)
        pg = compute_growth_probabilities(concentration, candidates, eta=1.0)
        choice_idx = rng.choice(len(candidate_indices), p=pg)
        cluster_mask[tuple(candidate_indices[choice_idx])] = True
    return cluster_mask, concentration, n_iters


t0 = time.perf_counter()
_, _, jac_iters = run_dla_jacobi(N=N, n_steps=N_STEPS_BENCH,
                                  rng=np.random.default_rng(SEED))
t_jac = time.perf_counter() - t0
print(f"  Jacobi vec: {t_jac:.2f} s   mean iters/step={np.mean(jac_iters):.0f}")
print(f"  Jacobi / SOR speed ratio: {t_jac/t_sor:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Wall-time per DLA step vs N for SOR warm-start
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Scaling: wall-time per DLA step vs N ===")
N_VALUES   = [30, 50, 75, 100]
N_SMALL    = 15   # steps per timing run
sor_times  = []
jac_times  = []
sor_niters = []
jac_niters = []

for Nv in N_VALUES:
    om = get_optimal_omega(Nv)
    t0 = time.perf_counter()
    _, _, si = run_dla(N=Nv, n_steps=N_SMALL, omega=om,
                        rng=np.random.default_rng(SEED))
    ts = (time.perf_counter() - t0) / N_SMALL
    t0 = time.perf_counter()
    _, _, ji = run_dla_jacobi(N=Nv, n_steps=N_SMALL,
                               rng=np.random.default_rng(SEED))
    tj = (time.perf_counter() - t0) / N_SMALL
    sor_times.append(ts); jac_times.append(tj)
    sor_niters.append(np.mean(si)); jac_niters.append(np.mean(ji))
    print(f"  N={Nv:3d}: SOR {ts:.3f}s ({np.mean(si):.0f} iters)  |  "
          f"Jacobi {tj:.3f}s ({np.mean(ji):.0f} iters)")


# ─── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1. Warm-start vs cold-start bar chart
ax = axes[0]
bars = ax.bar(["Cold start", "Warm start"],
              [r_cold.n_iter, r_warm.n_iter],
              color=["tomato", "steelblue"])
ax.set_ylabel("SOR iterations")
ax.set_title("Warm-start benefit\n(same cluster, N=100)")
for bar, val in zip(bars, [r_cold.n_iter, r_warm.n_iter]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha="center", va="bottom", fontsize=9)

# 2. SOR vs Jacobi timing bar chart
ax = axes[1]
ax.bar(["SOR (warm)", "Jacobi (vec)"],
       [t_sor, t_jac], color=["steelblue", "darkorange"])
ax.set_ylabel("Wall time (s)")
ax.set_title(f"SOR vs Jacobi\n({N_STEPS_BENCH} DLA steps, N=100)")

# 3. Scaling: time per step vs N
ax = axes[2]
ax.plot(N_VALUES, sor_times, "o-", label="SOR (Python loops)", color="steelblue")
ax.plot(N_VALUES, jac_times, "s-", label="Jacobi (numpy vectorised)", color="darkorange")
ax.set_xlabel("Grid size N")
ax.set_ylabel("Mean time per DLA step (s)")
ax.set_title("Scaling of BVP cost with N")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("DLA BVP solver performance  (SOR warm-start  vs  Jacobi vectorised)")
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / "a2_1_dla_performance.png"
plt.savefig(fname, dpi=150)
print(f"\nSaved → {fname}")
plt.show()
