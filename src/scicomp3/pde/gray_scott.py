"""Gray-Scott reaction-diffusion system (Assignment 2.3).

Models two reacting, diffusing chemicals U and V on a 2D periodic domain:

    ∂u/∂t = Dᵤ∇²u − uv² + f(1 − u)          (17)
    ∂v/∂t = D_v∇²v + uv² − (f + k)v           (18)

Discretisation:
    - Explicit (Forward Euler) time-stepping via the existing ``solve_ivp``.
    - 5-point finite difference Laplacian with periodic boundaries in both x
      and y (via ``np.roll``).
    - State vector shape: (N, N, 2) where ``state[..., 0] = u`` and
      ``state[..., 1] = v``.

Default parameters (assignment starting point):
    δt = 1,  δx = 1,  Dᵤ = 0.16,  D_v = 0.08,  f = 0.035,  k = 0.060

Stability:
    The FTCS explicit scheme requires 4δt·D/δx² ≤ 1.  With the given
    parameters and δx = δt = 1: 4·0.16·1/1² = 0.64 ≤ 1 ✓

References:
    J. E. Pearson, Science 261, 5118, pp. 189-192 (1993).
    Assignment Set 2, section 2.3 — The Gray-Scott model.
"""

import numpy as np


def gray_scott_rhs(t, state, Du, Dv, f, k, dx=1.0):
    """Compute the right-hand side of the Gray-Scott system.

    Args:
        t:     Current time (unused; kept for compatibility with solve_ivp).
        state: Array of shape *(N, N, 2)* where ``state[..., 0]`` is u and
               ``state[..., 1]`` is v.
        Du:    Diffusion coefficient for U.
        Dv:    Diffusion coefficient for V.
        f:     Feed rate (controls how fast U is replenished).
        k:     Kill rate (controls how fast V decays, net rate = f + k).
        dx:    Spatial grid spacing (default 1.0 as in the assignment).

    Returns:
        d_state: Array of the same shape as *state* containing
                 (∂u/∂t, ∂v/∂t) at every grid point.
    """
    u = state[..., 0]
    v = state[..., 1]

    # Discrete Laplacian with periodic BCs via np.roll
    inv_dx2 = 1.0 / (dx * dx)
    lap_u = inv_dx2 * (
        np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0)
        + np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1)
        - 4.0 * u
    )
    lap_v = inv_dx2 * (
        np.roll(v, -1, axis=0) + np.roll(v, 1, axis=0)
        + np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1)
        - 4.0 * v
    )

    uvv = u * v * v   # autocatalytic reaction term

    dudt = Du * lap_u - uvv + f * (1.0 - u)
    dvdt = Dv * lap_v + uvv - (f + k) * v

    return np.stack([dudt, dvdt], axis=-1)


def gray_scott_initial_conditions(N, square_size=None, noise_amplitude=0.05, rng=None):
    """Construct the default Gray-Scott initial conditions.

    - u = 1 + small noise everywhere
    - v = 0.25 + small noise inside a central square, 0 outside

    Args:
        N:               Grid points per side (square domain N×N).
        square_size:     Half-width (in grid points) of the central v-square.
                         Default: N // 10.
        noise_amplitude: Amplitude of random perturbations (default 0.05).
        rng:             NumPy ``Generator``.  If None, uses ``default_rng()``.

    Returns:
        state0: Float array of shape *(N, N, 2)*.
    """
    if rng is None:
        rng = np.random.default_rng()
    if square_size is None:
        square_size = max(2, N // 10)

    u = np.ones((N, N))
    v = np.zeros((N, N))

    # v = 0.25 in a small central square
    cx, cy = N // 2, N // 2
    half = square_size
    u[cx - half: cx + half, cy - half: cy + half] = 0.5
    v[cx - half: cx + half, cy - half: cy + half] = 0.25

    # Add small noise
    noise = noise_amplitude * rng.random((N, N, 2)) - noise_amplitude / 2
    state0 = np.stack([u, v], axis=-1) + noise
    state0 = np.clip(state0, 0.0, 1.0)

    return state0


def gray_scott_stable_dt(Du, Dv, dx=1.0, safety=0.9):
    """Compute a stable time step for the explicit Gray-Scott scheme.

    Stability condition (diffusion term dominant):
        4 δt D_max / δx² ≤ 1  →  δt ≤ δx² / (4 D_max)

    Args:
        Du, Dv:  Diffusion coefficients.
        dx:      Grid spacing.
        safety:  Safety factor (default 0.9).

    Returns:
        dt: Safe time step.
    """
    D_max = max(Du, Dv)
    return safety * dx**2 / (4.0 * D_max)
