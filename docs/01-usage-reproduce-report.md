# Assignment 3 — Usage & Reproduction Guide

## Prerequisites

```bash
cd assignment-submission
pip install -e ".[dev]"
pip install ngsolve   # required for FEM solver
```

Verify installation:

```bash
python -c "from scicomp3.pde.navier_stokes_fd import simulate_ns_fd; print('FD OK')"
python -c "from scicomp3.pde.lattice_boltzmann import simulate_lbm; print('LBM OK')"
python -c "from scicomp3.pde.navier_stokes_fem import simulate_ns_fem; print('FEM OK')"
python -c "from scicomp3.pde.helmholtz import solve_helmholtz_fd; print('Helmholtz OK')"
```

## Challenge A: Kármán Vortex Street (Navier-Stokes)

Three methods are implemented to simulate flow past a cylinder. All use the Schäfer-Turek benchmark geometry (channel 2.2 m × 0.41 m, cylinder at (0.2, 0.2) with r = 0.05 m).

### Method 1: Finite Difference (Projection Method)

```bash
MPLBACKEND=Agg python scripts/a3_1_ns_fd.py
```

- Solver: `src/scicomp3/pde/navier_stokes_fd.py`
- Chorin's projection method on a collocated grid
- Upwind advection, central-difference diffusion
- Pressure Poisson solved with `scipy.sparse.linalg.spsolve`
- Sweeps Re = {20, 50, 100, 200}
- Output: `images/figures/a3_1_ns_fd_vorticity.png`

### Method 2: Lattice Boltzmann (D2Q9 BGK)

```bash
MPLBACKEND=Agg python scripts/a3_1_ns_lbm.py
```

- Solver: `src/scicomp3/pde/lattice_boltzmann.py`
- D2Q9 lattice with BGK collision operator
- Bounce-back (no-slip), Zou-He inlet, extrapolation outlet
- Physical domain mapped to lattice units (D = 20 lattice nodes)
- Sweeps Re = {20, 50, 100, 200, 500}
- Output: `images/figures/a3_1_ns_lbm_vorticity.png`

### Method 3: Finite Element (ngsolve)

```bash
MPLBACKEND=Agg python scripts/a3_1_ns_fem.py
```

- Solver: `src/scicomp3/pde/navier_stokes_fem.py`
- Taylor-Hood elements (P2 velocity, P1 pressure) via ngsolve
- Semi-implicit time stepping (implicit diffusion, explicit convection)
- Unstructured triangular mesh with local refinement near cylinder
- Sweeps Re = {20, 100, 200}
- Output: `images/figures/a3_1_ns_fem_vorticity.png`

## Challenge B: WiFi Router Placement (Helmholtz)

```bash
MPLBACKEND=Agg python scripts/a3_2_wifi_optimize.py
```

- Solver: `src/scicomp3/pde/helmholtz.py`
- Helmholtz equation: Δu + k²u = f on a 10 m × 8 m floor plan
- Walls have complex refractive index (n = 2.5 + 0.5j → absorption)
- Impedance (absorbing) BC on outer walls
- Gaussian source at router position
- Wavenumber scaled by 1/3 for computational feasibility
- Grid search over candidate router positions
- Signal evaluated at 4 measurement points (avg |u|² in r = 5 cm disks)
- Output: `images/figures/a3_2_wifi_optimize.png`

## Source Code Structure

```
src/scicomp3/
├── core/result.py              # NSResult dataclass (added for A3)
├── pde/
│   ├── navier_stokes_fd.py     # FD projection method
│   ├── lattice_boltzmann.py    # LBM D2Q9 BGK
│   ├── navier_stokes_fem.py    # ngsolve FEM (Taylor-Hood)
│   └── helmholtz.py            # Helmholtz FD solver + floor plan
scripts/
├── a3_1_ns_fd.py               # FD vortex street sweep
├── a3_1_ns_lbm.py              # LBM vortex street sweep
├── a3_1_ns_fem.py              # FEM vortex street sweep
└── a3_2_wifi_optimize.py       # WiFi router optimization
```

## Quick Smoke Tests

Run individual solvers with minimal parameters to verify they work:

```bash
# FD: channel flow without cylinder (fast check)
python -c "
from scicomp3.pde.navier_stokes_fd import simulate_ns_fd
res = simulate_ns_fd(Nx=110, Ny=20, Lx=2.2, Ly=0.41, Re=20, U_in=1.0,
    D_cyl=0.1, cx=0.2, cy=0.2, r_cyl=0.05, T=1.0, save_every=500)
print(f'FD: stable={res.stable}, u_max={res.u[-1].max():.3f}')
"

# LBM: short run
python -c "
from scicomp3.pde.lattice_boltzmann import simulate_lbm
res = simulate_lbm(Nx=400, Ny=80, Re=50, U_in=0.02, cx=40, cy=40,
    r_cyl=20, D_cyl=40, n_steps=2000, save_every=1000)
print(f'LBM: stable={res.stable}, u_max={res.u[-1].max():.4f}')
"

# FEM: minimal run
python -c "
from scicomp3.pde.navier_stokes_fem import simulate_ns_fem
res = simulate_ns_fem(2.2, 0.41, 0.2, 0.2, 0.05, Re=20, U_in=1.0,
    D_cyl=0.1, T=0.05, dt=0.001, maxh=0.05, save_every=10)
print(f'FEM: stable={res.stable}, snapshots={len(res.t)}')
"

# Helmholtz: single position
python -c "
from scicomp3.pde.helmholtz import solve_helmholtz_fd, evaluate_signal_strength, MEASUREMENT_POINTS
u, x, y = solve_helmholtz_fd(2.4e9, 5.0, 4.0, Nx=100, Ny=80, k_scale=1/3)
_, total = evaluate_signal_strength(u, x, y, MEASUREMENT_POINTS)
print(f'Helmholtz: total_signal={total:.2e}')
"
```

## Known Limitations (Prototype)

- **FD solver**: Divergence/gradient near cylinder uses Python loops (slow). Needs vectorization or numba for grids > 200×50.
- **LBM**: Requires D ≥ 40 lattice nodes for Re > 50 (tau must stay well above 0.5).
- **FEM**: Sampling fields on a regular grid for output is slow (point-by-point evaluation). Consider using ngsolve's VTK export for post-processing.
- **Helmholtz floor plan**: Wall layout is approximate. Internal wall coordinates need verification against Figure 5 of the assignment.
- **WiFi optimization**: Grid search is brute-force. Consider Bayesian optimization or adjoint methods for faster convergence.
