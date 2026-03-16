"""Run KVS with finite differences and save an animation."""

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
Nx = 220
Ny = 41

N_STEPS = 10_000
PLOT_EVERY = N_STEPS // 200

# Default config matches the assignment spec exactly
config = KVSConfig(Re=Re, Nx=Nx, Ny=Ny)

# -- Capture frames via post_step callback -----------------------------------
frames = []


# -- Run simulation ----------------------------------------------------------
result = solve_kvs(config, method="fd", n_steps=N_STEPS, plot_every=PLOT_EVERY)
frames = result.snapshots
print(f"Simulation done. {len(frames)} frames captured.")

print(f"config.grid.shape: {config.grid.shape}")
print(f"frames[0].shape: {frames[0].shape}")


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
title = ax.set_title("KVS — FD solver — step 0")
plt.tight_layout()


def update(frame_idx):
    im.set_array(get_speed(frames[frame_idx]))
    title.set_text(f"KVS — FD solver — step {frame_idx * PLOT_EVERY}")
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
gif_path = out_dir / "a3_1_kvs_fd.gif"
print(f"Saving animation ({len(frames)} frames)...")
anim.save(gif_path, writer="pillow", fps=15, dpi=100)
print(f"Saved → {gif_path}")

plt.show()
