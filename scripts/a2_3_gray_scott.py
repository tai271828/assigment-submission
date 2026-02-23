"""Gray-Scott reaction-diffusion system (Assignment 2.3.E).

Simulates the Gray-Scott model on a 2D periodic domain using the parameters
from the assignment:

    δt=1, δx=1, Dᵤ=0.16, D_v=0.08, f=0.035, k=0.060

Plots the concentration of V at several time snapshots and produces a
side-by-side view of U and V at the final time.

Additional parameter sets are shown to illustrate the variety of patterns:
    - Spots:   f=0.035, k=0.060  (default)
    - Stripes: f=0.060, k=0.062
    - Labyrinthine: f=0.040, k=0.060
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots  # noqa: F401

styles = ["science"] if shutil.which("latex") else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.pde.gray_scott import (
    gray_scott_rhs,
    gray_scott_initial_conditions,
    gray_scott_stable_dt,
)
from scicomp3.ode.solver import solve_ivp

# ── Parameters ──────────────────────────────────────────────────────────────
N   = 100          # grid size N×N
DT  = 1.0          # time step (assignment default)
DX  = 1.0          # spatial step (assignment default)
Du  = 0.16
Dv  = 0.08
F   = 0.035
K   = 0.060
T_END       = 10_000   # long enough to see pattern formation
SAVE_EVERY  = 200      # save every 200 steps
SEED        = 0

# Verify stability
dt_stable = gray_scott_stable_dt(Du, Dv, DX)
print(f"Stable dt for diffusion: {dt_stable:.3f}  (using dt={DT} — "
      f"{'OK' if DT <= dt_stable else 'UNSTABLE!'})")

# ── Initial conditions ───────────────────────────────────────────────────────
rng    = np.random.default_rng(SEED)
state0 = gray_scott_initial_conditions(N, rng=rng)

print(f"Running Gray-Scott simulation: N={N}, T={T_END}, dt={DT}")
print(f"Parameters: Dᵤ={Du}, D_v={Dv}, f={F}, k={K}")

# ── Solve ────────────────────────────────────────────────────────────────────
result = solve_ivp(
    gray_scott_rhs,
    t_span=(0, T_END),
    y0=state0,
    method="forward_euler",
    dt=DT,
    args=(Du, Dv, F, K, DX),
    save_interval=SAVE_EVERY,
)

print(f"Saved {len(result.t)} snapshots")

# ── Plot snapshots of V ──────────────────────────────────────────────────────
snapshot_indices = [0, len(result.t) // 4, len(result.t) // 2,
                    3 * len(result.t) // 4, len(result.t) - 1]

fig, axes = plt.subplots(1, len(snapshot_indices), figsize=(4 * len(snapshot_indices), 4))

for ax, idx in zip(axes, snapshot_indices):
    v_field = result.y[idx, :, :, 1]
    im = ax.imshow(v_field.T, origin="lower", cmap="inferno",
                   vmin=0, vmax=0.4, aspect="equal")
    ax.set_title(f"$t = {result.t[idx]:.0f}$")
    ax.set_xlabel("$x$")
    if ax is axes[0]:
        ax.set_ylabel("$y$")
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axes[-1], label="$v$ concentration", fraction=0.046, pad=0.04)
fig.suptitle(f"Gray-Scott: concentration of $V$ over time\n"
             f"$D_u={Du}$, $D_v={Dv}$, $f={F}$, $k={K}$")
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / "a2_3_gray_scott_snapshots.png"
plt.savefig(fname, dpi=150)
print(f"Saved → {fname}")

# ── Final U and V side-by-side ────────────────────────────────────────────────
fig2, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(9, 4))
u_final = result.y[-1, :, :, 0]
v_final = result.y[-1, :, :, 1]

ax_u.imshow(u_final.T, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="equal")
ax_u.set_title(f"$u$ at $t={result.t[-1]:.0f}$")
ax_u.set_xticks([]); ax_u.set_yticks([])
ax_u.set_xlabel("$x$"); ax_u.set_ylabel("$y$")

im2 = ax_v.imshow(v_final.T, origin="lower", cmap="inferno", vmin=0, vmax=0.4, aspect="equal")
ax_v.set_title(f"$v$ at $t={result.t[-1]:.0f}$")
ax_v.set_xticks([]); ax_v.set_yticks([])
ax_v.set_xlabel("$x$")

fig2.colorbar(im2, ax=ax_v, label="concentration", fraction=0.046, pad=0.04)
fig2.suptitle("Gray-Scott: final concentration fields")
plt.tight_layout()

fname2 = out_dir / "a2_3_gray_scott_final.png"
plt.savefig(fname2, dpi=150)
print(f"Saved → {fname2}")
plt.show()
