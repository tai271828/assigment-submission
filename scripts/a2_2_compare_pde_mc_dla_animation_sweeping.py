"""Sweep equivalent (eta, Ps) pairs for PDE-DLA vs MC-DLA comparison.

Runs a2_2_compare_pde_mc_dla_animation.py for each pair of parameters that
produce morphologically similar clusters between the two DLA methods.

Rationale for equivalent pairs
------------------------------
Both eta (PDE-DLA) and Ps (MC-DLA) control cluster compactness:

  - eta controls how strongly growth concentrates at high-concentration tips.
    eta=0 → uniform (Eden, very compact). eta=1 → standard DLA.
    eta>1 → increasingly needle-like / open.

  - Ps controls how deeply walkers penetrate before sticking.
    Low Ps → walkers bounce many times, fill interior → compact.
    High Ps → walkers stick on first contact → branching (standard DLA).

The mapping is monotonic: more compact ←→ lower eta / lower Ps, and more
branching ←→ higher eta / higher Ps. There is no exact analytical mapping,
but empirically the following pairs produce clusters of comparable morphology:

  eta=0.0, Ps=0.05  — Very compact / Eden-like
  eta=0.5, Ps=0.2   — Moderately compact
  eta=1.0, Ps=1.0   — Standard DLA (both methods match by definition)
  eta=2.0, Ps=1.0   — Open / needle-like (MC cannot exceed Ps=1, so this
                       shows where the models diverge)
"""

import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).parent / "a2_2_compare_pde_mc_dla_animation.py"
PYTHON = sys.executable

N = 100
N_STEPS = 200
SEED = 42

# (eta, Ps) pairs — see docstring for rationale
PAIRS = [
    (0.0, 0.05),  # very compact / Eden-like
    (0.5, 0.2),   # moderately compact
    (1.0, 1.0),   # standard DLA
    (2.0, 1.0),   # open / needle-like (MC saturates at Ps=1)
]

print(f"Sweeping {len(PAIRS)} (eta, Ps) pairs on N={N} grid, {N_STEPS} steps each\n")

for i, (eta, ps) in enumerate(PAIRS, 1):
    print(f"{'='*60}")
    print(f"[{i}/{len(PAIRS)}] eta={eta}, Ps={ps}")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    cmd = [
        PYTHON, str(SCRIPT),
        "--N", str(N),
        "--eta", str(eta),
        "--ps", str(ps),
        "--steps", str(N_STEPS),
        "--seed", str(SEED),
    ]
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(cmd, env=env)

    elapsed = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{i}/{len(PAIRS)}] {status} in {elapsed:.1f}s\n")

    if result.returncode != 0:
        print("Aborting sweep due to failure.")
        sys.exit(1)

print("All sweeps completed.")
