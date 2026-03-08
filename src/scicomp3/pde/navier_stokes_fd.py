"""Finite-difference Navier-Stokes solver (Chorin projection, staggered grid).

Solves the 2D incompressible Navier-Stokes equations using the
projection method on a collocated grid with pressure correction.

Steps per time step:
    1. Predict velocity (advection + diffusion, explicit)
    2. Solve pressure Poisson: nabla^2(p) = (1/dt) * div(u*)
    3. Correct velocity: u^(n+1) = u* - dt * grad(p)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def create_cylinder_mask(Nx, Ny, Lx, Ly, cx, cy, r):
    """Create boolean mask for cells inside/on the cylinder."""
    dx = Lx / Nx
    dy = Ly / Ny
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r**2


def _build_pressure_laplacian(Nx, Ny, dx, dy, mask):
    """Build sparse Laplacian for pressure Poisson on collocated grid.

    Neumann BC on inlet/top/bottom/cylinder, Dirichlet (p=0) at outlet.
    """
    N = Nx * Ny
    diags = {0: np.zeros(N), 1: np.zeros(N - 1), -1: np.zeros(N - 1),
             Nx: np.zeros(N - Nx), -Nx: np.zeros(N - Nx)}

    idx = lambda i, j: j * Nx + i

    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j)
            if mask[j, i]:
                diags[0][k] = 1.0
                continue
            if i == Nx - 1:
                # Outlet: p = 0 (Dirichlet)
                diags[0][k] = 1.0
                continue

            coeff = 0.0
            # East neighbor
            if i < Nx - 1:
                if not mask[j, i + 1]:
                    diags[1][k] = 1.0 / dx**2
                    coeff -= 1.0 / dx**2
                else:
                    pass  # Neumann (skip)
            # West neighbor
            if i > 0:
                if not mask[j, i - 1]:
                    diags[-1][k - 1] = 1.0 / dx**2
                    coeff -= 1.0 / dx**2
            # else: inlet Neumann

            # North
            if j < Ny - 1:
                if not mask[j + 1, i]:
                    diags[Nx][k] = 1.0 / dy**2
                    coeff -= 1.0 / dy**2
            # South
            if j > 0:
                if not mask[j - 1, i]:
                    diags[-Nx][k - Nx] = 1.0 / dy**2
                    coeff -= 1.0 / dy**2

            diags[0][k] = coeff

    offsets = sorted(diags.keys())
    data = [diags[o] for o in offsets]
    # Build using diags constructor
    A = sparse.diags(data, offsets, shape=(N, N), format="csc")
    # Override specific rows for mask and outlet
    A = A.tolil()
    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j)
            if mask[j, i] or i == Nx - 1:
                A[k, :] = 0
                A[k, k] = 1.0
    return A.tocsc()


def simulate_ns_fd(
    Nx, Ny, Lx, Ly, Re, U_in, D_cyl, cx, cy, r_cyl,
    T, dt=None, save_every=100, verbose=True,
):
    """Run a finite-difference Navier-Stokes simulation (collocated grid).

    Args:
        Nx, Ny: Grid cells in x and y
        Lx, Ly: Domain size
        Re: Reynolds number
        U_in: Inlet velocity
        D_cyl: Cylinder diameter
        cx, cy: Cylinder center
        r_cyl: Cylinder radius
        T: Total simulation time
        dt: Time step (auto if None)
        save_every: Save interval
        verbose: Print progress

    Returns:
        NSResult
    """
    from ..core.result import NSResult

    dx = Lx / Nx
    dy = Ly / Ny
    nu = U_in * D_cyl / Re

    if dt is None:
        dt = min(0.2 * dx / U_in, 0.1 * dx**2 / nu)
    n_steps = int(T / dt)

    if verbose:
        print(f"NS-FD: Nx={Nx}, Ny={Ny}, Re={Re:.0f}, nu={nu:.6f}")
        print(f"  dx={dx:.4f}, dy={dy:.4f}, dt={dt:.6f}, steps={n_steps}")

    mask = create_cylinder_mask(Nx, Ny, Lx, Ly, cx, cy, r_cyl)

    # Build pressure Poisson
    A_p = _build_pressure_laplacian(Nx, Ny, dx, dy, mask)

    # Fields on collocated grid — initialize with uniform flow
    # (already divergence-free in interior, avoids impulsive start)
    u = np.full((Ny, Nx), U_in)
    v = np.zeros((Ny, Nx))
    p = np.zeros((Ny, Nx))
    u[mask] = 0.0
    u[0, :] = 0.0  # bottom wall
    u[-1, :] = 0.0  # top wall

    saved_t = [0.0]
    saved_u = [u.copy()]
    saved_v = [v.copy()]
    saved_p = [p.copy()]

    stable = True
    for step in range(1, n_steps + 1):
        t = step * dt

        # --- Advection-diffusion (explicit) ---
        # Pad for boundary conditions
        # u: inlet=U_in (west), zero-grad (east), no-slip (north/south)
        u_pad = np.pad(u, 1, mode="edge")
        u_pad[:, 0] = U_in       # inlet
        u_pad[0, :] = -u_pad[1, :]     # bottom no-slip (mirror)
        u_pad[-1, :] = -u_pad[-2, :]   # top no-slip (mirror)

        v_pad = np.pad(v, 1, mode="edge")
        v_pad[:, 0] = -v_pad[:, 1]     # inlet: v=0
        v_pad[0, :] = 0.0              # bottom no-slip
        v_pad[-1, :] = 0.0             # top no-slip

        # Slices for interior (maps to original grid)
        s = (slice(1, -1), slice(1, -1))

        # Central differences for diffusion
        d2udx2 = (u_pad[1:-1, 2:] - 2 * u_pad[s] + u_pad[1:-1, :-2]) / dx**2
        d2udy2 = (u_pad[2:, 1:-1] - 2 * u_pad[s] + u_pad[:-2, 1:-1]) / dy**2
        d2vdx2 = (v_pad[1:-1, 2:] - 2 * v_pad[s] + v_pad[1:-1, :-2]) / dx**2
        d2vdy2 = (v_pad[2:, 1:-1] - 2 * v_pad[s] + v_pad[:-2, 1:-1]) / dy**2

        # Upwind advection
        dudx_p = (u_pad[s] - u_pad[1:-1, :-2]) / dx  # backward
        dudx_m = (u_pad[1:-1, 2:] - u_pad[s]) / dx   # forward
        dudy_p = (u_pad[s] - u_pad[:-2, 1:-1]) / dy
        dudy_m = (u_pad[2:, 1:-1] - u_pad[s]) / dy

        dvdx_p = (v_pad[s] - v_pad[1:-1, :-2]) / dx
        dvdx_m = (v_pad[1:-1, 2:] - v_pad[s]) / dx
        dvdy_p = (v_pad[s] - v_pad[:-2, 1:-1]) / dy
        dvdy_m = (v_pad[2:, 1:-1] - v_pad[s]) / dy

        # Upwind selection
        adv_u = (np.maximum(u, 0) * dudx_p + np.minimum(u, 0) * dudx_m
                 + np.maximum(v, 0) * dudy_p + np.minimum(v, 0) * dudy_m)
        adv_v = (np.maximum(u, 0) * dvdx_p + np.minimum(u, 0) * dvdx_m
                 + np.maximum(v, 0) * dvdy_p + np.minimum(v, 0) * dvdy_m)

        u_star = u + dt * (nu * (d2udx2 + d2udy2) - adv_u)
        v_star = v + dt * (nu * (d2vdx2 + d2vdy2) - adv_v)

        # Enforce BCs on predicted velocity
        u_star[:, 0] = U_in
        u_star[:, -1] = u_star[:, -2]
        u_star[0, :] = 0.0
        u_star[-1, :] = 0.0
        u_star[mask] = 0.0

        v_star[:, 0] = 0.0
        v_star[:, -1] = v_star[:, -2]
        v_star[0, :] = 0.0
        v_star[-1, :] = 0.0
        v_star[mask] = 0.0

        # --- Pressure Poisson ---
        # div(u*) using forward differences (consistent with backward grad)
        # Skip faces adjacent to cylinder (Neumann BC: no flux through solid)
        div = np.zeros((Ny, Nx))
        # du*/dx (forward): skip if either cell is solid
        for j in range(Ny):
            for i in range(Nx - 1):
                if not mask[j, i] and not mask[j, i + 1]:
                    div[j, i] += (u_star[j, i + 1] - u_star[j, i]) / dx
        # dv*/dy (forward): skip if either cell is solid
        for j in range(Ny - 1):
            for i in range(Nx):
                if not mask[j, i] and not mask[j + 1, i]:
                    div[j, i] += (v_star[j + 1, i] - v_star[j, i]) / dy

        rhs = div.ravel() / dt
        # Zero RHS where p is prescribed (mask, outlet, inlet)
        rhs_2d = rhs.reshape(Ny, Nx)
        rhs_2d[mask] = 0.0
        rhs_2d[:, -1] = 0.0  # outlet
        rhs_2d[:, 0] = 0.0   # inlet (BC handles mass)
        rhs_2d[0, :] = 0.0   # bottom wall
        rhs_2d[-1, :] = 0.0  # top wall
        rhs = rhs_2d.ravel()

        p_flat = spsolve(A_p, rhs)
        p = p_flat.reshape(Ny, Nx)

        # --- Velocity correction (backward gradient, consistent with forward div) ---
        dpdx = np.zeros((Ny, Nx))
        for j in range(Ny):
            for i in range(1, Nx):
                if not mask[j, i] and not mask[j, i - 1]:
                    dpdx[j, i] = (p[j, i] - p[j, i - 1]) / dx

        dpdy = np.zeros((Ny, Nx))
        for j in range(1, Ny):
            for i in range(Nx):
                if not mask[j, i] and not mask[j - 1, i]:
                    dpdy[j, i] = (p[j, i] - p[j - 1, i]) / dy

        u = u_star - dt * dpdx
        v = v_star - dt * dpdy

        # Re-enforce BCs
        u[:, 0] = U_in
        u[:, -1] = u[:, -2]
        u[0, :] = 0.0
        u[-1, :] = 0.0
        u[mask] = 0.0

        v[:, 0] = 0.0
        v[:, -1] = v[:, -2]
        v[0, :] = 0.0
        v[-1, :] = 0.0
        v[mask] = 0.0

        # Stability check
        max_vel = max(np.abs(u).max(), np.abs(v).max())
        if np.isnan(max_vel) or max_vel > 20 * U_in:
            if verbose:
                print(f"  Blowup at step {step}, t={t:.4f}, max_vel={max_vel:.2f}")
            stable = False
            break

        if step % save_every == 0 or step == n_steps:
            saved_t.append(t)
            saved_u.append(u.copy())
            saved_v.append(v.copy())
            saved_p.append(p.copy())
            if verbose and step % (save_every * 10) == 0:
                print(f"  step {step}/{n_steps}, t={t:.4f}, max|u|={max_vel:.4f}")

    if verbose:
        print(f"  Done. stable={stable}, final t={saved_t[-1]:.4f}")

    return NSResult(
        t=np.array(saved_t),
        u=np.array(saved_u),
        v=np.array(saved_v),
        p=np.array(saved_p),
        Re=Re,
        method="fd",
        stable=stable,
    )
