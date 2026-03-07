"""
Functions to simulate the Gray-Scott model of reaction-diffusion systems, which
describes the interaction of two chemical species with diffusion and nonlinear reactions.

The model is given by the following PDEs:
    ∂u/∂t = D_u ∇²u - uv² + f(1-u)
    ∂v/∂t = D_v ∇²v + uv² - (f+k)v
where u and v are the concentrations of the two species, D_u and D_v are their diffusion
coefficients, and f and k are the feed and kill rates of the species.

The functions in this module implement both an explicit forward Euler scheme and an
implicit-explicit (IMEX) scheme for time-stepping the system, as well as a main function
to run the simulation over a specified time interval.
"""

import numpy as np

# ----------------------------------------------------------------
# Function for the explicit update scheme of the Gray-Scott model
# ----------------------------------------------------------------


def GS_fEuler_step(u, v, Du, Dv, f, k, dx, dt):
    """Performs one time step of the forward Euler scheme for the Gray-Scott model."""
    u_next = np.copy(u)
    v_next = np.copy(v)

    # add the horizontal and vertical diffusion terms according to the 5-point stencil for the Laplacian
    u_next += Du * dt / dx**2 * (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0))
    u_next += Du * dt / dx**2 * (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1))
    v_next += Dv * dt / dx**2 * (np.roll(v, -1, axis=0) - 2 * v + np.roll(v, 1, axis=0))
    v_next += Dv * dt / dx**2 * (np.roll(v, -1, axis=1) - 2 * v + np.roll(v, 1, axis=1))

    # add the reaction terms
    uvv = u * v * v  # nonlinear term
    Nu = -uvv + f * (1 - u)  # explicit term N_u(u,v)
    Nv = uvv - (f + k) * v  # explicit term N_v(u,v)

    u_next += dt * Nu
    v_next += dt * Nv

    return u_next, v_next


# ------------------------------------------------------------
# Functions for the IMEX update scheme of the Gray-Scott model
# ------------------------------------------------------------


def setup_FFT(N, dx):
    """Precompute the squared wavenumbers for the Fourier spectral method."""

    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    # Create 2D grids of squared wavenumbers
    kx2, ky2 = np.meshgrid(kx**2, ky**2, indexing="ij")
    # Compute the total squared wavenumber
    k2 = kx2 + ky2
    return k2


def GS_IMEX_euler_step(u, v, Du, Dv, f, k, dt, k2):
    """Performs one time step of the IMEX Euler scheme for the Gray-Scott model."""

    ## EXPLICIT PART OF THE UPDATE: compute the RHS according to forward Euler ##
    uvv = u * v * v  # nonlinear term
    Nu = -uvv + f * (1 - u)  # explicit term N_u(u,v)
    Nv = uvv - (f + k) * v  # explicit term N_v(u,v)

    RHS_u = u + dt * Nu
    RHS_v = v + dt * Nv

    ## IMPLICIT PART OF THE UPDATE: update in Fourier space according to hat(u^{n+1}) = A^{-1} * hat(bu) etc. ##
    # Fourier transform the right-hand side
    bu_hat = np.fft.fft2(RHS_u)
    bv_hat = np.fft.fft2(RHS_v)
    # Compute the update for the Fourier space coefficients of u and v
    denominator_u = 1 + dt * Du * k2
    denominator_v = 1 + dt * Dv * k2
    u_next_hat = bu_hat / denominator_u
    v_next_hat = bv_hat / denominator_v
    # Invert the Fourier transform to get the next u and v in real space
    u_next = np.real(np.fft.ifft2(u_next_hat))
    v_next = np.real(np.fft.ifft2(v_next_hat))

    return u_next, v_next, Nu, Nv


def GS_SBDF2_step(u, v, u_prev, v_prev, Nu_prev, Nv_prev, Du, Dv, f, k, dt, k2):
    """Performs one time step of the 2-SBDF scheme for the Gray-Scott model."""

    ## EXPLICIT PART OF THE UPDATE: compute the RHS according to forward Euler ##
    # Compute the net reaction at the current time step
    uvv = u * v * v  # nonlinear term
    Nu = -uvv + f * (1 - u)  # explicit term N_u(u,v)
    Nv = uvv - (f + k) * v  # explicit term N_v(u,v)
    # Construct the right-hand side for the implicit diffusion step
    rhs_u = (4 * u - u_prev) + 2 * dt * (2 * Nu - Nu_prev)
    rhs_v = (4 * v - v_prev) + 2 * dt * (2 * Nv - Nv_prev)

    ## IMPLICIT PART OF THE UPDATE: update in Fourier space according to hat(u^{n+1}) = A^{-1} * hat(bu) etc. ##
    # Fourier transform the right-hand side
    bu_hat = np.fft.fft2(rhs_u)
    bv_hat = np.fft.fft2(rhs_v)
    # Compute the update for the Fourier space coefficients of u and v
    denominator_u = 3 + 2 * dt * Du * k2
    denominator_v = 3 + 2 * dt * Dv * k2
    u_next_hat = bu_hat / denominator_u
    v_next_hat = bv_hat / denominator_v
    # Invert the Fourier transform to get the new u and v in real space
    u_next = np.real(np.fft.ifft2(u_next_hat))
    v_next = np.real(np.fft.ifft2(v_next_hat))

    return u_next, v_next, Nu, Nv


def simulate_Gray_Scott(
    u0, v0, Du, Dv, f, k, dx, dt, T_sim, step_method="IMEX", save_every=1
):
    """Simulate the discretized Gray-Scott model over time."""

    assert (
        4 * dt * max(Du, Dv) / dx**2 <= 1
    ), f"Stability condition not satisfied: increase N or decrease dt"

    u = np.copy(u0)
    v = np.copy(v0)
    N_steps = int(T_sim / dt)
    N_save = N_steps // save_every + 1
    time = np.arange(N_save) * save_every * dt

    # construct arrays to store the concentration fields at each time step
    u_states = np.zeros((N_save, *u.shape))
    u_states[0] = u.copy()  # store initial state
    v_states = np.zeros((N_save, *v.shape))
    v_states[0] = v.copy()  # store initial state

    if step_method == "IMEX":
        # Compute the discrete Fourier modes corresponding to the real-space grid
        k2 = setup_FFT(u.shape[0], dx)

        # take a first-order scheme to have enough data for the second-order SBDF2 scheme
        u_next, v_next, Nu, Nv = GS_IMEX_euler_step(u, v, Du, Dv, f, k, dt, k2)
        if save_every == 1:
            # save the computed next states
            u_states[1] = u_next.copy()
            v_states[1] = v_next.copy()
        stepfunc = GS_SBDF2_step
        stepfunc_params = [Du, Dv, f, k, dt, k2]

    elif step_method == "explicit":
        stepfunc = GS_fEuler_step
        stepfunc_params = [Du, Dv, f, k, dx, dt]
        # take the first step
        u, v = stepfunc(u, v, *stepfunc_params)
        if save_every == 1:
            # save the computed next states
            u_states[1] = u.copy()
            v_states[1] = v.copy()

    else:
        print(
            "Non-existing stepping method supplied. Choose 'IMEX' or 'explicit' instead."
        )
        return

    # evolve the system over time
    for t in range(2, N_steps + 1):
        if t % (N_steps // 10) == 0:  # print progress every 10% of the simulation
            print(f"Progress: {t/N_steps:.1%}")

        if step_method == "IMEX":
            u_prev, v_prev = u, v
            u, v = u_next, v_next
            Nu_prev, Nv_prev = Nu, Nv
            u_next, v_next, Nu, Nv = stepfunc(
                u, v, u_prev, v_prev, Nu_prev, Nv_prev, *stepfunc_params
            )
        else:
            u, v = stepfunc(u, v, *stepfunc_params)

        if t % save_every != 0:
            continue  # skip saving intermediate states to save memory

        # save the computed next states
        u_states[t // save_every] = u.copy()
        v_states[t // save_every] = v.copy()

    return (
        time,
        np.transpose(u_states, (0, 2, 1)),
        np.transpose(v_states, (0, 2, 1)),
    )  # Transpose to (time, x, y) for easier plotting
