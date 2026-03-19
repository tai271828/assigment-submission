"""Run KVS with lattice Boltzmann and save an animation."""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from scicomp3.core.config import KVSConfig
from scicomp3.kvs.solver import solve_kvs

import scienceplots  # noqa: F401

styles = (
    ["science"]
    if (shutil.which("latex") and shutil.which("dvipng"))
    else ["science", "no-latex"]
)
plt.style.use(styles)

# -- Parameters --------------------------------------------------------------
Re = 150

# LBM needs a finer grid than FD for stability at Re=150.
# D_lb (cylinder diameter in lattice units) should be >= 20 so that
# tau = 3*nu_lb + 0.5 stays comfortably above 0.5.
# With Nx=440, Ny=82: dx=dy=0.005 m, D_lb=20, tau≈0.54.
Nx = 440
Ny = 82

N_STEPS = 10_000
PLOT_EVERY = N_STEPS // 200

config = KVSConfig(Re=Re, Nx=Nx, Ny=Ny)

# -- Run simulation ----------------------------------------------------------
result = solve_kvs(
    config, method="lb", n_steps=N_STEPS, plot_every=PLOT_EVERY, U_lb=0.1
)
frames = result.snapshots
print(f"Simulation done. {len(frames)} frames captured.")


def get_speed(frame):
    """Compute velocity magnitude from a (2, Nx+1, Ny+1) frame."""
    ux, uy = frame[0], frame[1]
    speed = np.sqrt(ux**2 + uy**2)
    speed[config.cylinder_mask] = np.nan
    return speed.T


# -- Build animation ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))

im = ax.imshow(
    get_speed(frames[0]),
    origin="lower",
    cmap="viridis",
    vmin=0,
    vmax=config.U_inlet * 2,
    aspect="auto",
    extent=[0, config.Lx, 0, config.Ly],
)
fig.colorbar(im, ax=ax, label="Velocity magnitude")
ax.set_xlabel("$x$ [m]")
ax.set_ylabel("$y$ [m]")
title = ax.set_title("KVS — LBM solver — step 0")
plt.tight_layout()


def update(frame_idx):
    im.set_array(get_speed(frames[frame_idx]))
    title.set_text(f"KVS — LBM solver — step {frame_idx * PLOT_EVERY}")
    return im, title


anim = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=50,
    blit=False,
    repeat=True,
)

# -- Save --------------------------------------------------------------------
out_dir = Path(__file__).parent.parent / "images" / "gifs"
out_dir.mkdir(parents=True, exist_ok=True)
gif_path = out_dir / "a3_1_kvs_lb.gif"
print(f"Saving animation ({len(frames)} frames)...")
anim.save(gif_path, writer="pillow", fps=15, dpi=100)
print(f"Saved → {gif_path}")

plt.show()
