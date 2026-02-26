"""DLA performance and scaling analysis (Assignment 2.1.B).

Compares DLA solve time for different grid sizes and demonstrates the
benefit of warm-starting SOR from the previous concentration field.
Also compares cold-start vs warm-start iteration counts.
"""

import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots  # noqa: F401

styles = ["science"] if (shutil.which("latex") and shutil.which("dvipng")) else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.models.dla import run_dla
from scicomp3.bvp.omega import get_optimal_omega

# ── Parameters ──────────────────────────────────────────────────────────────
GRID_SIZES = [50, 75, 100, 150]
N_STEPS = 100
ETA = 1.0
SEED = 42

print("DLA Performance Scaling Analysis")
print("=" * 50)
print(f"Growth steps per run: {N_STEPS}, η={ETA}\n")

# ── Run simulations for different grid sizes ────────────────────────────────
perf_results = {}
for N in GRID_SIZES:
    omega = get_optimal_omega(N)
    rng = np.random.default_rng(SEED)
    print(f"  N={N:>3d}  (ω={omega:.4f}) …", end=" ", flush=True)

    t_start = time.perf_counter()
    cluster_mask, concentration, n_iters = run_dla(
        N=N, n_steps=N_STEPS, eta=ETA, omega=omega, rng=rng
    )
    elapsed = time.perf_counter() - t_start

    total_iters = sum(n_iters)
    print(f"done in {elapsed:.1f}s  "
          f"(total SOR iters={total_iters:,}, "
          f"mean/step={np.mean(n_iters):.1f})")
    perf_results[N] = {
        "elapsed": elapsed,
        "total_iters": total_iters,
        "mean_iters": np.mean(n_iters),
        "n_iters": n_iters,
        "n_cluster": cluster_mask.sum(),
    }

# ── Warm-start vs cold-start comparison (N=100) ────────────────────────────
print("\n--- Warm-start vs cold-start comparison (N=100) ---")
N_COMPARE = 100
N_STEPS_COMPARE = 50
omega = get_optimal_omega(N_COMPARE)

# Warm start (default — as implemented)
rng = np.random.default_rng(SEED)
t0 = time.perf_counter()
_, _, iters_warm = run_dla(
    N=N_COMPARE, n_steps=N_STEPS_COMPARE, eta=ETA, omega=omega, rng=rng
)
t_warm = time.perf_counter() - t0
print(f"  Warm-start: {t_warm:.2f}s, total iters={sum(iters_warm):,}, mean={np.mean(iters_warm):.1f}")

# Cold start — reset concentration to linear gradient each step
from scicomp3.bvp.solver import solve_bvp
from scicomp3.models.dla import find_growth_candidates, compute_growth_probabilities, _bvp_bc

rng = np.random.default_rng(SEED)
cluster_mask_cold = np.zeros((N_COMPARE + 1, N_COMPARE + 1), dtype=bool)
cluster_mask_cold[N_COMPARE // 2, 1] = True
iters_cold = []
t0 = time.perf_counter()
for _ in range(N_STEPS_COMPARE):
    # Cold start: always begin from linear gradient
    y_coords = np.linspace(0.0, 1.0, N_COMPARE + 1)
    c_cold = np.broadcast_to(y_coords, (N_COMPARE + 1, N_COMPARE + 1)).copy()
    sink_coords = np.argwhere(cluster_mask_cold)
    result = solve_bvp(
        c_cold, method="sor", tol=1e-4, max_iter=2_000,
        post_step=_bvp_bc, sink_coordinates=sink_coords, omega=omega,
    )
    iters_cold.append(result.n_iter)
    candidates = find_growth_candidates(cluster_mask_cold)
    if not np.any(candidates):
        break
    candidate_indices = np.argwhere(candidates)
    pg = compute_growth_probabilities(result.y, candidates, ETA)
    choice_idx = rng.choice(len(candidate_indices), p=pg)
    new_point = tuple(candidate_indices[choice_idx])
    cluster_mask_cold[new_point] = True
t_cold = time.perf_counter() - t0
print(f"  Cold-start: {t_cold:.2f}s, total iters={sum(iters_cold):,}, mean={np.mean(iters_cold):.1f}")
print(f"  Warm-start speedup: {sum(iters_cold)/sum(iters_warm):.1f}× fewer total iters")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1. Wall-clock time vs grid size
ax = axes[0]
Ns = sorted(perf_results.keys())
times = [perf_results[n]["elapsed"] for n in Ns]
ax.plot(Ns, times, "o-", markersize=6)
ax.set_xlabel("Grid size $N$")
ax.set_ylabel("Wall-clock time (s)")
ax.set_title(f"DLA solve time vs grid size\n({N_STEPS} growth steps)")
ax.grid(True, alpha=0.3)

# 2. Mean SOR iterations per step vs grid size
ax = axes[1]
means = [perf_results[n]["mean_iters"] for n in Ns]
ax.plot(Ns, means, "s-", markersize=6, color="C1")
ax.set_xlabel("Grid size $N$")
ax.set_ylabel("Mean SOR iters per step")
ax.set_title("Mean SOR iterations vs grid size")
ax.grid(True, alpha=0.3)

# 3. Warm-start vs cold-start iteration comparison
ax = axes[2]
steps = np.arange(1, N_STEPS_COMPARE + 1)
ax.plot(steps, iters_warm, lw=0.8, label="Warm-start", alpha=0.8)
ax.plot(steps, iters_cold, lw=0.8, label="Cold-start", alpha=0.8)
ax.set_xlabel("Growth step")
ax.set_ylabel("SOR iterations")
ax.set_title("Warm-start vs cold-start\nSOR iterations per step")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("DLA Performance Analysis (Assignment 2.1.B)")
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / "a2_1_dla_perf.png"
plt.savefig(fname, dpi=150)
print(f"\nSaved → {fname}")
plt.show()
