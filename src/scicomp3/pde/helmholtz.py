"""Helmholtz equation solver for WiFi signal propagation.

Solves: nabla^2(u) + k^2 * u = f
on a 2D floor plan with walls (complex refractive index).

Boundary conditions:
    - Outer walls: impedance (absorbing) BC: du/dn - iku = 0
    - Internal walls: continuity with complex k (absorption)

The source f is a Gaussian pulse at the router position.
"""

import numpy as np


# --------------------------------------------------------------------------
# Floor plan geometry (from assignment Figure 5)
# House: 10m x 8m, wall thickness 0.15m
# --------------------------------------------------------------------------

HOUSE_LX = 10.0
HOUSE_LY = 8.0
WALL_THICKNESS = 0.15

# Wall segments as (x_start, y_start, x_end, y_end)
# Outer walls are handled by boundary conditions.
# Internal walls (horizontal and vertical segments):
INTERNAL_WALLS = [
    # Vertical wall between living room and kitchen/bathroom area
    (3.0, 0.0, 3.0, 4.0),
    # Horizontal wall separating lower rooms
    (3.0, 4.0, 7.0, 4.0),
    # Vertical wall between kitchen and bathroom
    (7.0, 0.0, 7.0, 4.0),
    # Horizontal wall at mid-height (right side)
    (7.0, 4.0, 10.0, 4.0),
    # Vertical wall between bedrooms
    (7.0, 4.0, 7.0, 8.0),
]


def create_floor_plan_mask(Nx, Ny, Lx=HOUSE_LX, Ly=HOUSE_LY, t=WALL_THICKNESS):
    """Create a boolean mask for wall regions on a uniform grid.

    Args:
        Nx, Ny: Grid points in x and y
        Lx, Ly: Domain size
        t: Wall thickness

    Returns:
        wall_mask: boolean array (Ny, Nx), True in wall regions
    """
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    wall = np.zeros((Ny, Nx), dtype=bool)
    ht = t / 2  # half thickness

    for x1, y1, x2, y2 in INTERNAL_WALLS:
        if x1 == x2:
            # Vertical wall
            mask = (np.abs(X - x1) <= ht) & (Y >= min(y1, y2)) & (Y <= max(y1, y2))
        else:
            # Horizontal wall
            mask = (np.abs(Y - y1) <= ht) & (X >= min(x1, x2)) & (X <= max(x1, x2))
        wall |= mask

    return wall


def solve_helmholtz_fd(
    freq_hz,
    router_x,
    router_y,
    Nx=400,
    Ny=320,
    Lx=HOUSE_LX,
    Ly=HOUSE_LY,
    n_air=1.0,
    n_wall=2.5 + 0.5j,
    A_source=1e4,
    sigma_source=0.2,
    k_scale=1.0,
):
    """Solve the Helmholtz equation on the floor plan using finite differences.

    Uses a 5-point stencil: nabla^2(u) + k^2(x,y)*u = f(x,y)
    with impedance BC on outer walls and heterogeneous k in wall regions.

    Args:
        freq_hz: WiFi frequency in Hz (e.g. 2.4e9)
        router_x, router_y: Router position
        Nx, Ny: Grid resolution
        Lx, Ly: Domain size
        n_air: Refractive index of air
        n_wall: Complex refractive index of walls
        A_source: Source amplitude
        sigma_source: Source Gaussian width
        k_scale: Scale factor for wavenumber (1.0 = full, 1/3 = scaled)

    Returns:
        u: Complex solution field (Ny, Nx)
        x: x coordinates
        y: y coordinates
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    c = 3e8  # speed of light
    k0 = 2 * np.pi * freq_hz / c * k_scale  # free-space wavenumber (possibly scaled)

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Wall mask
    wall_mask = create_floor_plan_mask(Nx, Ny, Lx, Ly)

    # Wavenumber field (complex in walls)
    k_field = np.where(wall_mask, k0 * n_wall, k0 * n_air)

    # Source term
    r2 = (X - router_x) ** 2 + (Y - router_y) ** 2
    source = A_source * np.exp(-r2 / (2 * sigma_source**2))

    # Build sparse system
    N = Nx * Ny
    A = lil_matrix((N, N), dtype=complex)
    b = np.zeros(N, dtype=complex)

    def idx(i, j):
        return j * Nx + i

    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j)
            k2 = k_field[j, i] ** 2

            if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                # Impedance BC: du/dn - ik*u = 0
                # Discretize: for left boundary (i=0):
                #   (u[1,j] - u[0,j])/dx - ik*u[0,j] = 0
                #   => u[0,j]*(1/dx + ik) = u[1,j]/dx
                # Similar for other boundaries
                ik = 1j * k_field[j, i]
                A[k, k] = 1.0

                if i == 0:
                    A[k, k] = -1.0 / dx - ik
                    A[k, idx(1, j)] = 1.0 / dx
                elif i == Nx - 1:
                    A[k, k] = -1.0 / dx - ik
                    A[k, idx(Nx - 2, j)] = 1.0 / dx
                elif j == 0:
                    A[k, k] = -1.0 / dy - ik
                    A[k, idx(i, 1)] = 1.0 / dy
                elif j == Ny - 1:
                    A[k, k] = -1.0 / dy - ik
                    A[k, idx(i, Ny - 2)] = 1.0 / dy

                b[k] = 0.0
            else:
                # Interior: nabla^2(u) + k^2*u = -f
                # (u[i+1,j] + u[i-1,j] - 2u[i,j])/dx^2
                # + (u[i,j+1] + u[i,j-1] - 2u[i,j])/dy^2 + k^2*u = -f
                A[k, idx(i + 1, j)] = 1.0 / dx**2
                A[k, idx(i - 1, j)] = 1.0 / dx**2
                A[k, idx(i, j + 1)] = 1.0 / dy**2
                A[k, idx(i, j - 1)] = 1.0 / dy**2
                A[k, k] = -2.0 / dx**2 - 2.0 / dy**2 + k2
                b[k] = -source[j, i]

    print(f"  Solving Helmholtz: {N} unknowns, k0={k0:.2f} rad/m...")
    A_csc = A.tocsc()
    u_flat = spsolve(A_csc, b)
    u = u_flat.reshape((Ny, Nx))

    return u, x, y


def evaluate_signal_strength(u, x, y, measurement_points, r_avg=0.05):
    """Evaluate signal strength at measurement points.

    Signal strength at each point is the average of |u|^2 in a circle
    of radius r_avg around the point.

    Args:
        u: Complex field (Ny, Nx)
        x, y: Coordinate arrays
        measurement_points: list of (x, y) tuples
        r_avg: Averaging radius (default 5 cm)

    Returns:
        signals: list of signal strengths at each point
        total: sum of all signal strengths
    """
    X, Y = np.meshgrid(x, y)
    intensity = np.abs(u) ** 2
    signals = []

    for mx, my in measurement_points:
        dist2 = (X - mx) ** 2 + (Y - my) ** 2
        mask = dist2 <= r_avg**2
        if mask.sum() > 0:
            signals.append(intensity[mask].mean())
        else:
            # Point is too close to a wall or outside; use nearest value
            j = np.argmin(np.abs(y - my))
            i = np.argmin(np.abs(x - mx))
            signals.append(intensity[j, i])

    return signals, sum(signals)


# Measurement points from the assignment
MEASUREMENT_POINTS = [
    (1.0, 5.0),  # Living room
    (2.0, 1.0),  # Kitchen
    (9.0, 1.0),  # Bathroom
    (9.0, 7.0),  # Bedroom 1
]
