"""
Assignment 1.1 - Smoke Test

Quick test to verify the wave equation solver works.
Uses sin(5πx) initial condition.
"""

import numpy as np
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use("TkAgg")

from scicomp3 import solve_ivp, wave1d_rhs


# Parameters
c = 1
L = 1
N = 90  # number of intervals (N+1 grid points)
dt = 1e-3
T_sim = 10

# Setup grid with N+1 points to include both boundaries
x = np.linspace(0, L, N + 1)  # [0, dx, 2dx, ..., L]

# Initial condition: sin(5πx) with boundary conditions enforced
psi0 = np.sin(5 * np.pi * x)
psi0[0] = 0  # enforce boundary condition at x=0
psi0[-1] = 0  # enforce boundary condition at x=L
v0 = np.zeros(N + 1)
y0 = np.column_stack([psi0, v0])


def fixed_ends(t, y):
    """Enforce fixed boundary conditions: psi=0 at both ends."""
    y[0, 0] = 0
    y[-1, 0] = 0
    return y


# Solve
result = solve_ivp(
    wave1d_rhs,
    t_span=(0, T_sim),
    y0=y0,
    method="symplectic_euler",
    dt=dt,
    args=(c, L, N + 1),
    post_step=fixed_ends,
)

print(f"Integration: {result.message}")
print(f"Function evaluations: {result.nfev}")

# Extract amplitudes
amplitudes = result.y[:, :, 0]

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
for i in range(0, len(result.t), len(result.t) // 11):
    ax.plot(x, amplitudes[i], color=plt.cm.cividis(i / len(result.t)))

ax.set_xlabel("Position along string (x)")
ax.set_ylabel("String amplitude (Ψ)")
ax.set_title("Vibrating string over time")

cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap="cividis"),
    ax=ax,
    label="Time",
    ticks=np.linspace(0, 1, 3),
)
cbar.ax.set_yticklabels([f"{t:.1f}" for t in np.linspace(0, T_sim, 3)])

plt.tight_layout()
plt.show()
