"""
Assignment 2.3 - Plotting Patterns in the Gray-Scott Model

Solves the 2D reaction-diffusion equation:
    ∂u/∂t = D_u ∇²u - uv² + f(1-u)
    ∂v/∂t = D_v ∇²v + uv² - (f+k)v

Boundary conditions:
    periodic in x and y directions

Initial condition:
    u(x, y, t=0) = 0.5 for 0 ≤ x ≤ 1, 0 ≤ y < 1
    v(x, y, t=0) = 0 for 0 ≤ x ≤ 1, 0 ≤ y < 1
    with a small square in the center where v = 0.25,
    plus small random noise to break symmetry

Tasks: Plot the concentration fields u and v at several times to visualize the patterns that emerge.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import scienceplots
import shutil

mpl.use("TkAgg")
plt.rcParams.update({"font.size": 13})
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10

from scicomp3.models.gray_scott import simulate_Gray_Scott

styles = (
    ["science"]
    if (shutil.which("latex") and shutil.which("dvipng"))
    else ["science", "no-latex"]
)
plt.style.use(styles)

# Output directory
output_dir = Path(__file__).parent.parent / "images" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(0)

N = 256  # grid size (number of grid points)

# INITIAL CONDITIONS
initial_conditions = "assignment"
# initial_conditions = "pearson"
if initial_conditions == "assignment":
    # Assignment initial conditions: u=0.5, v=0 everywhere, except small square with v=0.25
    u0 = np.ones((N, N)) * 0.5  # initial concentration field
    v0 = np.zeros((N, N))  # initial concentration field
    v0[5 * N // 12 : 7 * N // 12, 5 * N // 12 : 7 * N // 12] = (
        0.25  # set small nonzero concentration at the center
    )
elif initial_conditions == "pearson":
    # Pearson's initial conditions: u=1, v=0 everywhere, except small square with u=0.5, v=0.25
    u0 = np.ones((N, N)) * 1  # initial concentration field
    v0 = np.zeros((N, N))  # initial concentration field
    u0[5 * N // 12 : 7 * N // 12, 5 * N // 12 : 7 * N // 12] = (
        0.5  # set small nonzero concentration at the center
    )
    v0[5 * N // 12 : 7 * N // 12, 5 * N // 12 : 7 * N // 12] = (
        0.25  # set small nonzero concentration at the center
    )
# Add some small random noise to break symmetry and trigger pattern formation
u0 += np.random.rand(N, N) * 0.05  # add small random noise to break symmetry
v0 += np.random.rand(N, N) * 0.05  # add small random noise to break symmetry

# PARAMETER SETUP
assignment_params = [
    0.16,
    0.08,
    0.035,
    0.060,
    1,
]  # Assignment parameter values, correspond to theta (class 2-a)
pearson_params = [2e-5, 1e-5, 0.035, 0.060, 2.5 / N]  # Pearsons parameter values
delta_params = [
    0.16,
    0.08,
    0.030,
    0.055,
    1,
]  # arranged spots parameter values (class 2-a)
epsilon_params = [
    0.16,
    0.08,
    0.022,
    0.059,
    1,
]  # chaotic spots parameter values (class 3)
alpha_params = [
    0.16,
    0.08,
    0.010,
    0.047,
    1,
]  # wavelets parameter values (class 3)
rho_params = [
    0.16,
    0.08,
    0.090,
    0.059,
    1,
]  # giraffes parameter values (class 2-a)

# Choose parameter values
Du, Dv, f, k, dx = delta_params

dt = 1  # time step size
T_sim = 1000 * dt  # total simulation time

method = "explicit"
# method = "IMEX"

# Run the simulation
save_every = 500  # save every n-th time step to reduce memory usage
time, u_states, v_states = simulate_Gray_Scott(
    u0, v0, Du, Dv, f, k, dx, dt, T_sim, step_method=method, save_every=save_every
)


# Plotting
# indices_to_plot = [0, len(u_states) // 2, -1]
indices_to_plot = [-1]

fig, axs = plt.subplots(
    2,
    len(indices_to_plot),
    figsize=(len(indices_to_plot) * 2 + 1, 4),
    sharex=True,
    sharey=True,
)
for i in range(len(indices_to_plot)):
    ax_u = axs[0, i] if len(indices_to_plot) != 1 else axs[0]
    ax_v = axs[1, i] if len(indices_to_plot) != 1 else axs[1]
    idx = indices_to_plot[i]
    imu = ax_u.imshow(
        u_states[idx],
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="inferno",
        vmin=np.min(u_states),
        vmax=np.max(u_states),
    )
    imv = ax_v.imshow(
        v_states[idx],
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="cividis",
        vmin=np.min(v_states),
        vmax=np.max(v_states),
    )
    ax_u.set_title(
        f"$t=$ {idx*save_every*dt:.1f}" if idx != -1 else f"$t=$ {T_sim:.1f}"
    )
    ax_u.set_xlabel(r"$x$ [a.u.]")
    ax_u.set_xticks([0, 0.5, 1])
    ax_v.set_xlabel(r"$x$ [a.u.]")
    ax_u.set_yticks([0, 0.5, 1])
    ax_v.set_yticks([0, 0.5, 1])

if len(indices_to_plot) == 1:
    axs[0].set_ylabel(r"$y$ [a.u.]")
    axs[1].set_ylabel(r"$y$ [a.u.]")
else:
    axs[0, 0].set_ylabel(r"$y$ [a.u.]")
    axs[1, 0].set_ylabel(r"$y$ [a.u.]")
fig.subplots_adjust(right=0.8)
cbar_ax_u = fig.add_axes(
    [0.72 + 0.03 * len(indices_to_plot), 0.533, 0.06 / len(indices_to_plot), 0.345]
)
fig.colorbar(imu, cax=cbar_ax_u, label="conc. $u$")
cbar_ax_v = fig.add_axes(
    [0.72 + 0.03 * len(indices_to_plot), 0.113, 0.06 / len(indices_to_plot), 0.345]
)
fig.colorbar(imv, cax=cbar_ax_v, label="conc. $v$")

filename = (
    output_dir
    / f"a3_2_patterns_{method}_N={N}_Du={Du}_Dv={Dv}_f={f}_k={k}_dx={dx}_dt={dt}_Tsim={T_sim}.png"
)
plt.savefig(filename, dpi=300, bbox_inches="tight")
print("Saved: ", filename)

plt.show()
