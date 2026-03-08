"""Finite Element Navier-Stokes solver using ngsolve.

Solves the 2D incompressible Navier-Stokes equations on an unstructured
triangular mesh with Taylor-Hood (P2/P1) elements for velocity/pressure.

Time integration uses a semi-implicit scheme:
    - Diffusion: implicit (backward Euler)
    - Convection: explicit (linearized with previous velocity)
    - Pressure: implicit (saddle-point system)

Requires: ngsolve, netgen
"""

import numpy as np


def create_channel_mesh(Lx, Ly, cx, cy, r, maxh=0.05, maxh_cyl=None):
    """Create a 2D channel mesh with a circular cylinder.

    Args:
        Lx, Ly: Channel dimensions
        cx, cy: Cylinder center
        r: Cylinder radius
        maxh: Maximum element size (far field)
        maxh_cyl: Maximum element size near cylinder (default: maxh/3)

    Returns:
        ngsolve.Mesh
    """
    from netgen.geom2d import SplineGeometry

    if maxh_cyl is None:
        maxh_cyl = maxh / 3

    geo = SplineGeometry()

    # Outer rectangle (channel)
    # Points: bottom-left, bottom-right, top-right, top-left
    p1 = geo.AppendPoint(0, 0)
    p2 = geo.AppendPoint(Lx, 0)
    p3 = geo.AppendPoint(Lx, Ly)
    p4 = geo.AppendPoint(0, Ly)

    # Channel boundary segments with boundary names
    geo.Append(["line", p1, p2], leftdomain=1, rightdomain=0, bc="wall_bottom")
    geo.Append(["line", p2, p3], leftdomain=1, rightdomain=0, bc="outlet")
    geo.Append(["line", p3, p4], leftdomain=1, rightdomain=0, bc="wall_top")
    geo.Append(["line", p4, p1], leftdomain=1, rightdomain=0, bc="inlet")

    # Cylinder (as circle, hole in domain)
    geo.AddCircle(c=(cx, cy), r=r, leftdomain=0, rightdomain=1, bc="cylinder",
                  maxh=maxh_cyl)

    geo.SetMaterial(1, "fluid")

    mesh = geo.GenerateMesh(maxh=maxh)

    import ngsolve
    return ngsolve.Mesh(mesh)


def simulate_ns_fem(
    Lx,
    Ly,
    cx,
    cy,
    r_cyl,
    Re,
    U_in,
    D_cyl,
    T,
    dt,
    maxh=0.05,
    save_every=10,
    order=2,
    verbose=True,
):
    """Run an ngsolve FEM Navier-Stokes simulation.

    Uses Taylor-Hood elements (P{order}/P{order-1}) with semi-implicit
    time stepping.

    Args:
        Lx, Ly: Channel dimensions
        cx, cy: Cylinder center
        r_cyl: Cylinder radius
        Re: Reynolds number
        U_in: Inlet velocity
        D_cyl: Cylinder diameter (for Re)
        T: Simulation time
        dt: Time step
        maxh: Mesh element size
        save_every: Save every N steps
        order: Polynomial order for velocity (pressure is order-1)
        verbose: Print progress

    Returns:
        NSResult with velocity/pressure snapshots on a regular grid
    """
    import ngsolve as ngs
    from ..core.result import NSResult

    nu = U_in * D_cyl / Re
    n_steps = int(T / dt)

    if verbose:
        print(f"NS-FEM: Re={Re:.0f}, nu={nu:.6f}, dt={dt:.5f}, steps={n_steps}")
        print(f"  Channel: {Lx}x{Ly}, cylinder at ({cx},{cy}), r={r_cyl}")

    # Create mesh
    mesh = create_channel_mesh(Lx, Ly, cx, cy, r_cyl, maxh=maxh)
    if verbose:
        print(f"  Mesh: {mesh.ne} elements, {mesh.nv} vertices")

    # Function spaces: Taylor-Hood
    V = ngs.VectorH1(mesh, order=order, dirichlet="inlet|wall_top|wall_bottom|cylinder")
    Q = ngs.H1(mesh, order=order - 1)
    X = ngs.FESpace([V, Q])

    # Trial and test functions
    (u, p), (v, q) = X.TnT()

    # Bilinear form: implicit diffusion + pressure coupling
    a = ngs.BilinearForm(X)
    a += (1 / dt) * ngs.InnerProduct(u, v) * ngs.dx
    a += nu * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v)) * ngs.dx
    a += (-1) * p * ngs.div(v) * ngs.dx
    a += (-1) * q * ngs.div(u) * ngs.dx
    a.Assemble()

    # Grid function for solution
    gfu = ngs.GridFunction(X)
    velocity, pressure = gfu.components

    # Set initial/boundary conditions
    inlet_vel = ngs.CoefficientFunction((U_in, 0))
    velocity.Set(inlet_vel, definedon=mesh.Boundaries("inlet"))
    velocity.Set(ngs.CoefficientFunction((0, 0)),
                 definedon=mesh.Boundaries("wall_top|wall_bottom|cylinder"))

    # Previous velocity for convection linearization
    gfu_old = ngs.GridFunction(X)
    vel_old = gfu_old.components[0]

    # Linear form (RHS): previous time step + explicit convection
    f_form = ngs.LinearForm(X)

    # Solver
    inv = a.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

    # Sampling grid for output
    Nx_out, Ny_out = 200, 80
    x_out = np.linspace(0, Lx, Nx_out)
    y_out = np.linspace(0, Ly, Ny_out)

    saved_t = [0.0]
    u_init, v_init, p_init = _sample_fields(mesh, velocity, pressure, x_out, y_out)
    saved_u = [u_init]
    saved_v = [v_init]
    saved_p = [p_init]

    stable = True
    for step in range(1, n_steps + 1):
        t = step * dt

        # Copy current solution as "old"
        gfu_old.vec.data = gfu.vec

        # Build RHS: (1/dt)*u_old - (u_old · grad)u_old
        f_form = ngs.LinearForm(X)
        f_form += (1 / dt) * ngs.InnerProduct(vel_old, v) * ngs.dx
        conv = ngs.InnerProduct(ngs.Grad(vel_old) * vel_old, v) * ngs.dx
        f_form += (-1) * conv
        f_form.Assemble()

        # Solve
        res = f_form.vec - a.mat * gfu.vec
        gfu.vec.data += inv * res

        # Re-apply Dirichlet BCs
        velocity.Set(inlet_vel, definedon=mesh.Boundaries("inlet"))
        velocity.Set(ngs.CoefficientFunction((0, 0)),
                     definedon=mesh.Boundaries("wall_top|wall_bottom|cylinder"))

        # Stability check
        max_vel = _max_velocity(mesh, velocity)
        if np.isnan(max_vel) or max_vel > 50 * U_in:
            if verbose:
                print(f"  Blowup at step {step}, t={t:.4f}")
            stable = False
            break

        if step % save_every == 0 or step == n_steps:
            u_s, v_s, p_s = _sample_fields(mesh, velocity, pressure, x_out, y_out)
            saved_t.append(t)
            saved_u.append(u_s)
            saved_v.append(v_s)
            saved_p.append(p_s)
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
        method="fem",
        stable=stable,
    )


def _sample_fields(mesh, velocity, pressure, x_out, y_out):
    """Sample velocity and pressure on a regular grid."""
    import ngsolve as ngs

    Nx = len(x_out)
    Ny = len(y_out)
    u_arr = np.zeros((Ny, Nx))
    v_arr = np.zeros((Ny, Nx))
    p_arr = np.zeros((Ny, Nx))

    for j, yy in enumerate(y_out):
        for i, xx in enumerate(x_out):
            try:
                mp = mesh(xx, yy)
                vel = velocity(mp)
                u_arr[j, i] = vel[0]
                v_arr[j, i] = vel[1]
                p_arr[j, i] = pressure(mp)
            except Exception:
                pass  # outside mesh (inside cylinder)

    return u_arr, v_arr, p_arr


def _max_velocity(mesh, velocity):
    """Estimate maximum velocity magnitude."""
    import ngsolve as ngs

    vel_norm = ngs.sqrt(ngs.InnerProduct(velocity, velocity))
    try:
        return max(abs(vel_norm(mesh(x, y)))
                   for x in np.linspace(0.1, 2.0, 20)
                   for y in np.linspace(0.05, 0.36, 10))
    except Exception:
        return 0.0
