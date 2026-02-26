"""DLA with different ω values (Assignment 2.1.A — ω optimisation).

Tests whether the theoretical optimal ω for SOR still works well when
the DLA cluster is present as a sink.  Runs a fixed number of DLA growth
steps with several ω values and compares total SOR iteration count.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots  # noqa: F401

styles = ["science"] if (shutil.which("latex") and shutil.which("dvipng")) else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.models.dla import run_dla
from scicomp3.bvp.omega import get_optimal_omega

# ── Parameters ──────────────────────────────────────────────────────────────
N = 100
N_STEPS = 150
ETA = 1.0
SEED = 42
OMEGA_THEORETICAL = get_optimal_omega(N)

# Test a range of ω values around the theoretical optimum
OMEGA_VALUES = [1.0, 1.3, 1.5, 1.6, 1.7, 1.8, 1.85, 1.9, OMEGA_THEORETICAL]
OMEGA_VALUES = sorted(set(round(w, 4) for w in OMEGA_VALUES))

print(f"Theoretical optimal ω = {OMEGA_THEORETICAL:.4f}")
print(f"Testing ω ∈ {OMEGA_VALUES}")
print(f"DLA: N={N}, n_steps={N_STEPS}, η={ETA}\n")

# ── Run simulations ─────────────────────────────────────────────────────────
results = {}
for omega in OMEGA_VALUES:
    rng = np.random.default_rng(SEED)  # same seed → same growth sequence
    print(f"  ω={omega:.4f} …", end=" ", flush=True)
    cluster_mask, concentration, n_iters = run_dla(
        N=N, n_steps=N_STEPS, eta=ETA, omega=omega, rng=rng
    )
    total = sum(n_iters)
    mean = np.mean(n_iters)
    print(f"total SOR iters = {total:,}, mean/step = {mean:.1f}")
    results[omega] = {
        "n_iters": n_iters,
        "total": total,
        "mean": mean,
    }

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 1. Total SOR iterations vs ω
ax = axes[0]
omegas = sorted(results.keys())
totals = [results[w]["total"] for w in omegas]
ax.plot(omegas, totals, "o-", markersize=5)
ax.axvline(OMEGA_THEORETICAL, color="C3", ls="--", lw=0.8,
           label=f"$\\omega_{{opt}}$ (theory) = {OMEGA_THEORETICAL:.4f}")
best_omega = min(results, key=lambda w: results[w]["total"])
ax.axvline(best_omega, color="C2", ls=":", lw=0.8,
           label=f"$\\omega_{{best}}$ (DLA) = {best_omega:.4f}")
ax.set_xlabel("$\\omega$")
ax.set_ylabel("Total SOR iterations")
ax.set_title(f"Total SOR cost vs $\\omega$\n({N_STEPS} DLA steps)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 2. Mean iterations per step vs ω
ax = axes[1]
means = [results[w]["mean"] for w in omegas]
ax.bar(range(len(omegas)), means, tick_label=[f"{w:.2f}" for w in omegas],
       color=["C3" if abs(w - OMEGA_THEORETICAL) < 0.001 else "C0" for w in omegas])
ax.set_xlabel("$\\omega$")
ax.set_ylabel("Mean SOR iterations per step")
ax.set_title("Mean SOR iterations per growth step")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle(f"SOR $\\omega$ optimisation in DLA (N={N})")
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / "a2_1_dla_omega.png"
plt.savefig(fname, dpi=150)
print(f"\nSaved → {fname}")
plt.show()
