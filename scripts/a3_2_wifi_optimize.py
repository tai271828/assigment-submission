"""WiFi router placement optimization (Assignment 3.2).

Solves the Helmholtz equation for different router positions on the
floor plan and finds the position maximizing total signal strength
at 4 measurement points.

Uses a scaled wavenumber (k/3) for computational tractability.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from scicomp3.pde.helmholtz import (
    solve_helmholtz_fd,
    evaluate_signal_strength,
    create_floor_plan_mask,
    MEASUREMENT_POINTS,
    HOUSE_LX,
    HOUSE_LY,
)

# -- Parameters ---------------------------------------------------------------
FREQ = 2.4e9  # 2.4 GHz
K_SCALE = 1.0 / 3.0  # scale wavenumber for feasibility
NX, NY = 200, 160  # grid resolution
ROUTER_GRID_STEP = 0.5  # coarse search step [m]
ROUTER_REFINE_STEP = 0.2  # fine search step [m]

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# -- Phase 1: Coarse grid search ---------------------------------------------
wall_mask = create_floor_plan_mask(NX, NY, HOUSE_LX, HOUSE_LY)

# Generate candidate router positions (only in air regions)
x_grid = np.arange(0.3, HOUSE_LX - 0.3, ROUTER_GRID_STEP)
y_grid = np.arange(0.3, HOUSE_LY - 0.3, ROUTER_REFINE_STEP)

# Check which positions are in air
x_arr = np.linspace(0, HOUSE_LX, NX)
y_arr = np.linspace(0, HOUSE_LY, NY)

candidates = []
for rx in x_grid:
    for ry in y_grid:
        ix = np.argmin(np.abs(x_arr - rx))
        iy = np.argmin(np.abs(y_arr - ry))
        if not wall_mask[iy, ix]:
            candidates.append((rx, ry))

print(f"Coarse search: {len(candidates)} candidate positions")
print(f"Frequency: {FREQ/1e9:.1f} GHz, k_scale={K_SCALE}")
print(f"Grid: {NX}x{NY}")
print()

best_signal = -np.inf
best_pos = None
all_signals = []

for idx, (rx, ry) in enumerate(candidates):
    t0 = time.perf_counter()
    u, x, y = solve_helmholtz_fd(
        FREQ, rx, ry, Nx=NX, Ny=NY, k_scale=K_SCALE,
    )
    signals, total = evaluate_signal_strength(u, x, y, MEASUREMENT_POINTS)
    elapsed = time.perf_counter() - t0

    all_signals.append((rx, ry, total, signals))
    if total > best_signal:
        best_signal = total
        best_pos = (rx, ry)

    if (idx + 1) % 10 == 0 or idx == 0:
        print(f"  [{idx+1}/{len(candidates)}] ({rx:.1f}, {ry:.1f}) "
              f"total={total:.2e}  ({elapsed:.1f}s)")

print(f"\nBest position: ({best_pos[0]:.1f}, {best_pos[1]:.1f})")
print(f"Best total signal: {best_signal:.2e}")

# -- Plot signal map for best position ----------------------------------------
print("\nComputing signal map at optimal position...")
u_best, x, y = solve_helmholtz_fd(
    FREQ, best_pos[0], best_pos[1], Nx=NX, Ny=NY, k_scale=K_SCALE,
)
signals_best, total_best = evaluate_signal_strength(u_best, x, y, MEASUREMENT_POINTS)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: signal strength (|u|^2) in dB
X, Y = np.meshgrid(x, y)
intensity = np.abs(u_best) ** 2
intensity_db = 10 * np.log10(intensity + 1e-30)

ax = axes[0]
im = ax.pcolormesh(X, Y, intensity_db, cmap="hot", shading="auto")
fig.colorbar(im, ax=ax, label="Signal strength [dB]")

# Overlay walls
ax.contour(X, Y, wall_mask.astype(float), levels=[0.5], colors="white", linewidths=1)

# Mark router and measurement points
ax.plot(best_pos[0], best_pos[1], 'c*', ms=15, label="Router")
room_names = ["Living room", "Kitchen", "Bathroom", "Bedroom 1"]
for (mx, my), name, sig in zip(MEASUREMENT_POINTS, room_names, signals_best):
    ax.plot(mx, my, 'go', ms=8)
    ax.annotate(f"{name}\n{sig:.1e}", (mx, my), color="lime", fontsize=7,
                ha="center", va="bottom", textcoords="offset points", xytext=(0, 5))

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"WiFi signal at ({best_pos[0]:.1f}, {best_pos[1]:.1f}), "
             f"f={FREQ/1e9:.1f} GHz (k/{int(1/K_SCALE)})")
ax.set_aspect("equal")
ax.legend(loc="upper right")

# Right: optimization landscape
ax2 = axes[1]
rx_vals = sorted(set(s[0] for s in all_signals))
ry_vals = sorted(set(s[1] for s in all_signals))
signal_map = np.full((len(ry_vals), len(rx_vals)), np.nan)
for rx, ry, total, _ in all_signals:
    ix = rx_vals.index(rx)
    iy = ry_vals.index(ry)
    signal_map[iy, ix] = total

RX, RY = np.meshgrid(rx_vals, ry_vals)
im2 = ax2.pcolormesh(RX, RY, np.log10(signal_map + 1e-30), cmap="viridis",
                     shading="auto")
fig.colorbar(im2, ax=ax2, label="log10(total signal)")
ax2.contour(X, Y, wall_mask.astype(float), levels=[0.5], colors="white",
            linewidths=0.5)
ax2.plot(best_pos[0], best_pos[1], 'r*', ms=15, label="Optimal")
for mx, my in MEASUREMENT_POINTS:
    ax2.plot(mx, my, 'go', ms=6)
ax2.set_xlabel("Router x [m]")
ax2.set_ylabel("Router y [m]")
ax2.set_title("Optimization landscape")
ax2.set_aspect("equal")
ax2.legend()

plt.tight_layout()
plt.savefig(out_dir / "a3_2_wifi_optimize.png", dpi=150)
print(f"Saved to {out_dir / 'a3_2_wifi_optimize.png'}")

# -- Summary ------------------------------------------------------------------
print(f"\n--- Results ---")
print(f"Router position: ({best_pos[0]:.1f}, {best_pos[1]:.1f}) m")
print(f"Total signal: {total_best:.2e}")
for (mx, my), name, sig in zip(MEASUREMENT_POINTS, room_names, signals_best):
    print(f"  {name} ({mx}, {my}): {sig:.2e}")

plt.show()
