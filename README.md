# scicomp3 — Scientific Computing Assignment Package

Numerical solvers for the 1D wave equation, 2D diffusion equation, steady-state Laplace equation, and Diffusion Limited Aggregation (DLA), built for the Scientific Computing course (Assignment Sets 1 & 2).

## Quick Start

`scicomp3` is released to [PyPI](https://test.pypi.org/project/scicomp3/). You can easily to use it by installing via pip.

```bash
pip install scicomp3
```

Now you are ready to try everything under `scripts`!!


For example, test the installation:

```bash
# Clone and enter the project
cd assigment-submission

# Run tests
pytest tests/ -v

# Run a script
python scripts/a1_1_smoke_test.py
```


### Development

Assuming you are using `pip`, you can develop and test the code as a package locally by the below setup.

```bash
# Clone and enter the project
cd assigment-submission

# Create/activate venv and install (editable mode with dev deps)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run a script
python scripts/a1_1_smoke_test.py
```

## Project Structure

```
.
├── src/scicomp3/              # Main package
│   ├── core/
│   │   ├── grid.py            # Grid1D, Grid2D — spatial discretization
│   │   └── result.py          # ODEResult, BVPResult, DLASORResult, DLAMCResult — solver output containers
│   ├── ode/
│   │   ├── methods.py         # Time-stepping: Euler, symplectic Euler
│   │   └── solver.py          # solve_ivp() — IVP solver entry point
│   ├── pde/
│   │   ├── wave.py            # wave1d_rhs, initial conditions (cases i–iii), analytical solution
│   │   └── diffusion.py       # diffusion2d_rhs, BCs, stable dt, analytical solution
│   ├── bvp/
│   │   ├── methods.py         # Iterative methods: Jacobi, Gauss-Seidel, SOR
│   │   ├── methods_numba.py   # Numba JIT-compiled SOR kernel (~100x speedup)
│   │   ├── solver.py          # solve_bvp() — BVP solver entry point
│   │   └── omega.py           # Optimal omega computation and search for SOR
│   ├── objects/
│   │   ├── shapes.py          # Geometric coordinate generation (rectangles, circles)
│   │   ├── sink.py            # Sink region utilities (Dirichlet, c=0)
│   │   └── insulator.py       # Insulator region utilities (Neumann, zero-flux)
│   ├── models/
│   │   ├── dla_by_sor.py      # grow_dla_sor() — DLA via steady-state diffusion (SOR)
│   │   ├── dla_by_mc.py       # grow_dla_mc() — DLA via Monte Carlo random walkers
│   │   └── gray_scott.py      # Gray-Scott reaction-diffusion model (IMEX Euler, FFT)
│   └── analysis/
│       └── cluster.py         # ClusterStats, fractal dimension (box-counting)
│
├── tests/                     # Pytest test suite
│   ├── test_boundary_conditions.py  # Wave BC enforcement (3 cases)
│   ├── test_diffusion.py           # Diffusion solver + analytical comparison
│   ├── test_jacobi.py              # Jacobi iteration convergence + steady state
│   ├── test_gauss_seidel.py        # Gauss-Seidel iteration convergence
│   ├── test_sor.py                 # SOR iteration with omega = 1.9
│   ├── test_a2_1.py                # DLA growth correctness + reference comparison
│   └── test_scripts.py             # Smoke tests for all scripts
│
├── scripts/                   # Runnable plotting/animation/benchmark scripts
│   ├── a1_1_*.py                   # Wave equation (smoke test, plots, animations, analytical)
│   ├── a1_2_*.py                   # 2D diffusion (snapshots, animation, verification)
│   ├── a1_6_*.py                   # Iterative methods, omega, sinks, insulators
│   ├── a2_1_*.py                   # DLA by SOR (plots, animation, eta sweep, benchmarks)
│   ├── a2_2_*.py                   # DLA comparison SOR vs MC (plots, animations, sweeps)
│   └── a2_3_*.py                   # Pattern analysis
│
├── contrib/                   # C benchmark code for SOR kernel comparison
├── data/                      # Cached simulation data (e.g. n_vs_omega.pkl, DLA references)
├── pyproject.toml             # Build config (hatchling), deps, pytest settings
└── images/                    # Generated figures and GIFs
```

## How It Works

### Solving a PDE (Initial Value Problem)

The package separates concerns into three layers:

1. **PDE right-hand side** (`pde/`) — defines the physics (spatial derivatives)
2. **ODE solver** (`ode/`) — advances the solution in time
3. **Post-step callback** — enforces boundary conditions after each step

Example: solving the wave equation end-to-end:

```python
import numpy as np
from scicomp3 import Grid1D, solve_ivp, wave1d_rhs
from scicomp3.pde.wave import initial_condition_case_ii

# 1. Grid and parameters
grid = Grid1D(N=90, L=1.0)
c, dt, T_sim = 1.0, 1e-3, 2.0

# 2. Initial condition: Ψ(x,0) = sin(5πx), Ψ_t(x,0) = 0
psi0 = initial_condition_case_ii(grid.x)
psi0[0] = psi0[-1] = 0          # enforce BCs on IC
v0 = np.zeros(grid.N)
y0 = np.column_stack([psi0, v0]) # state = [Ψ, v], shape (N, 2)

# 3. Boundary condition callback
def fixed_ends(t, y):
    y[0, 0] = y[-1, 0] = 0
    return y

# 4. Solve
result = solve_ivp(
    wave1d_rhs,
    t_span=(0, T_sim),
    y0=y0,
    method="symplectic_euler",    # energy-preserving for wave eq
    dt=dt,
    args=(c, grid.L, grid.N),
    post_step=fixed_ends,
)

# 5. Extract results
amplitudes = result.y[:, :, 0]   # Ψ at each saved time step
times = result.t
```

Example: 2D diffusion equation:

```python
import numpy as np
from scicomp3.core.grid import Grid2D
from scicomp3.ode.solver import solve_ivp
from scicomp3.pde.diffusion import (
    diffusion2d_rhs, apply_diffusion_bc, diffusion_stable_dt, analytical_solution,
)

grid = Grid2D(N=50, L=1.0)
D = 1.0
dt = diffusion_stable_dt(D, grid.dx)  # auto-compute safe dt

c0 = np.zeros((grid.N + 1, grid.N + 1))
apply_diffusion_bc(c0)                # set top=1, bottom=0

result = solve_ivp(
    diffusion2d_rhs,
    t_span=(0, 1.0),
    y0=c0,
    method="forward_euler",
    dt=dt,
    args=(D, grid.dx),
    post_step=lambda t, y: (apply_diffusion_bc(y), y)[1],
    save_interval=100,
)
```

### Solving a BVP (Steady-State)

For steady-state problems (Laplace equation), iterative solvers are available:

```python
from scicomp3.bvp.solver import solve_bvp
from scicomp3.bvp.omega import get_optimal_omega

result = solve_bvp(c0, method="sor", post_step=fixed_bc, tol=1e-5,
                   omega=get_optimal_omega(N))
# result.y       — converged solution
# result.n_iter  — iterations to convergence
# result.delta_history — convergence measure per iteration
```

### Available Time-Stepping Methods

| Name | Aliases | Order | Properties |
|------|---------|-------|------------|
| `"symplectic_euler"` | `"euler_cromer"`, `"semi_implicit_euler"` | 1st | Symplectic (energy-preserving). Default. Use for wave equation. |
| `"forward_euler"` | `"euler"` | 1st | Explicit. Use for diffusion equation. |

### Available Iterative Methods (BVP)

| Name | Description |
|------|-------------|
| `"jacobi"` | Jacobi iteration — vectorized with `np.roll` |
| `"gauss_seidel"` | Gauss-Seidel — sequential updates using latest values |
| `"sor"` | Successive Over-Relaxation — accelerated Gauss-Seidel with relaxation parameter omega |
| `"sor_numba"` | Numba JIT-compiled SOR — same algorithm as `"sor"`, ~100x faster via `@njit` |

Methods are registered in `scicomp3.ode.methods.METHODS` (IVP) and `scicomp3.bvp.methods` (BVP), looked up by name in `solve_ivp()` and `solve_bvp()`.

### Key Design Decisions

**Post-step callback for boundary conditions.** The PDE RHS functions use `np.roll` for the spatial stencil, which wraps boundary values incorrectly. The `post_step` callback in `solve_ivp` corrects this after every time step. This separates the physics from the constraints cleanly.

**Symplectic Euler for wave equation.** Forward Euler is dissipative — it loses energy over time. The symplectic Euler method preserves the phase-space structure of Hamiltonian systems, keeping energy bounded over long simulations. This is the default method.

**Explicit scheme for diffusion.** The FTCS (Forward Time, Centered Space) scheme is simple but conditionally stable. Use `diffusion_stable_dt()` to compute a safe time step (90% of the theoretical maximum δx² / 4D).

**Sink and insulator objects.** The BVP solvers support sink (Dirichlet, c=0) and insulator (Neumann, zero-flux) objects via coordinate arrays passed to `solve_bvp()`.

### Diffusion Limited Aggregation (DLA)

Two DLA methods are available: PDE-based (via SOR) and Monte Carlo (via random walkers).

**PDE-based DLA** (`grow_dla_sor`) alternates between solving the steady-state diffusion equation via SOR and growing the aggregate by one point. Growth probability is proportional to c^eta:

```python
from scicomp3.models.dla_by_sor import grow_dla_sor
from scicomp3.bvp.omega import get_optimal_omega

N = 50
omega = get_optimal_omega(N)
result = grow_dla_sor(
    n_iter_growth=100,
    growth_seed=(N // 2, 1),
    eta=1.0,
    y0=c0,
    omega=omega,
    tol=1e-4,
    max_iter_sor=2_000,
    post_step=fixed_bc,        # enforce BCs each SOR iteration
    post_growth=my_callback,   # optional: called after each growth step
    method="sor_numba",        # use numba-accelerated solver
)
# result.y                  — final concentration field
# result.growth_mask        — boolean mask of cluster sites
# result.bvp_iters_per_step — BVP iteration count for each growth step
```

The `eta` parameter controls cluster shape: low eta gives bushy growth, high eta gives spindly branches.

**Monte Carlo DLA** (`grow_dla_mc`) uses random walkers that spawn at the top boundary and stick to the aggregate with probability `sticking_probability`:

```python
from scicomp3.models.dla_by_mc import grow_dla_mc

result = grow_dla_mc(
    n_walking_steps=50_000,
    growth_seed=(N // 2, 1),
    sticking_probability=1.0,
    N=N,
)
# result.growth_mask — boolean mask of cluster sites
```

### Gray-Scott Reaction-Diffusion

The Gray-Scott model simulates pattern formation via two coupled reaction-diffusion equations. Supports Forward Euler and IMEX (spectral) time-stepping:

```python
from scicomp3.models.gray_scott import GS_IMEX_euler_step, setup_FFT
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_diffusion.py -v

# With print output
pytest tests/ -v -s
```

The test suite covers:
- **test_boundary_conditions.py** — Wave equation BCs hold for all 3 initial conditions; documents np.roll boundary pollution without post_step.
- **test_diffusion.py** — Diffusion BCs (top=1, bottom=0), convergence to steady state c=y, match against analytical erfc series solution, stability check.
- **test_jacobi.py** — Jacobi iteration convergence, steady-state profile, boundary checks, monotonicity.
- **test_gauss_seidel.py** — Gauss-Seidel convergence, fewer iterations than Jacobi.
- **test_sor.py** — SOR iteration with omega=1.9.
- **test_a2_1.py** — DLA growth correctness and comparison against reference data.
- **test_scripts.py** — Smoke tests that every script under `scripts/` runs without error.

## Running Scripts

Scripts live in `scripts/` and produce plots, animations, or benchmark output. They require `matplotlib` and `scienceplots`:

```bash
# --- Assignment 1.1: Wave equation ---
python scripts/a1_1_smoke_test.py                    # Quick smoke test
python scripts/a1_1_cases_plot.py                     # Static plots for 3 ICs
python scripts/a1_1_cases_animation.py                # Animated GIFs
python scripts/a1_1_cases_compared_to_analytical.py   # Numerical vs analytical error

# --- Assignment 1.2: 2D diffusion ---
python scripts/a1_2_diffusion.py                      # Concentration field snapshots
python scripts/a1_2_diffusion_animation.py             # Animated GIF
python scripts/a1_2_diffusion_verification.py          # Verification vs analytical solution

# --- Assignment 1.6: Iterative methods ---
python scripts/a1_6_iterative_methods.py              # Compare Jacobi/GS/SOR
python scripts/a1_6_iterative_convergence.py          # Convergence rate comparison
python scripts/a1_6_iterative_jacobi.py               # Jacobi standalone
python scripts/a1_6_iterative_gauss_seidel.py         # Gauss-Seidel standalone
python scripts/a1_6_iterative_sor.py                  # SOR standalone
python scripts/a1_6_objects_k_impact.py               # Object impact on iterations
python scripts/a1_6_seeking_optimal_omega.py          # Optimal omega search
python scripts/a1_6_omega_values.py                   # Omega parameter exploration
python scripts/a1_6_omega_for_various_N_sim.py        # Omega vs grid size simulation
python scripts/a1_6_omega_for_various_N_plot.py       # Omega vs grid size plotting
python scripts/a1_6_sinks_jacobi.py                   # Sink with Jacobi
python scripts/a1_6_sinks_gauss_seidel.py             # Sink with Gauss-Seidel
python scripts/a1_6_sinks_sor.py                      # Sink with SOR
python scripts/a1_6_sinks_sor_animation.py            # Sink SOR animation
python scripts/a1_6_sinks_k_impact.py                 # Sink impact on convergence
python scripts/a1_6_sinks_and_insulators_sor.py       # Combined sink + insulator
python scripts/a1_6_insulators_jacobi.py              # Insulator with Jacobi
python scripts/a1_6_insulators_gauss_seidel.py        # Insulator with Gauss-Seidel
python scripts/a1_6_insulators_sor.py                 # Insulator with SOR
python scripts/a1_6_insulators_sor_animation.py       # Insulator SOR animation
python scripts/a1_6_insulators_k_impact.py            # Insulator impact on convergence

# --- Assignment 2.1: DLA by SOR ---
python scripts/a2_1_dla_by_sor.py                     # DLA cluster (static plot)
python scripts/a2_1_dla_by_sor_animation.py           # DLA growth animation (GIF)
python scripts/a2_1_dla_eta_sweep.py                  # Cluster shape vs eta
python scripts/a2_1_dla_by_sor_numba.py               # Numba-accelerated DLA
python scripts/a2_1_dla_by_sor_benchmark.py           # Benchmark: solver x init x omega (8 cases)
python scripts/a2_1_dla_by_sor_benchmark_numba_warmup.py          # Benchmark: numba cold vs warm JIT
python scripts/a2_1_dla_by_sor_benchmark_interpretation_numba.py  # Numba performance interpretation
python scripts/a2_1_dla_by_sor_numba_benchmark_diff_N.py          # Benchmark: numba scaling with N

# --- Assignment 2.2: DLA comparison SOR vs MC ---
python scripts/a2_2_dla_comparison_sor_vs_mc_sim.py   # Simulate both SOR and MC methods
python scripts/a2_2_dla_comparison_sor_vs_mc_plot.py  # Plot comparison
python scripts/a2_2_compare_pde_mc_dla_animation.py   # Side-by-side animation (3 panels)
python scripts/a2_2_compare_pde_mc_dla_animation_sweeping.py  # Sweep (eta, Ps) pairs
python scripts/a2_2_dla_by_mc_animation.py            # MC DLA animation
python scripts/a2_2_dla_by_mc_animation_legacy.py     # MC DLA animation (legacy API)
python scripts/a2_2_dla_by_mc_sticking_probabilities.py  # Sticking probability analysis

# --- Assignment 2.3: Pattern analysis ---
python scripts/a2_3_plot_patterns.py                  # Plot cluster patterns
```

## Adding a New Time-Stepping Method

1. Define a step function in `src/scicomp3/ode/methods.py`:

```python
def my_step(fun, t, y, dt, args=()):
    """One step of my method. Returns (y_new, n_function_evals)."""
    dydt = fun(t, y, *args)
    y_new = ...  # your update rule
    return y_new, 1
```

2. Register it in `METHODS`:

```python
METHODS["my_method"] = my_step
```

3. Use it:

```python
result = solve_ivp(..., method="my_method")
```

## Adding a New PDE

1. Create a RHS function in `src/scicomp3/pde/`:

```python
def my_pde_rhs(t, y, *params):
    """Compute dy/dt for my PDE. Returns array same shape as y."""
    ...
```

2. Write a post_step if boundary conditions need enforcement.
3. Call `solve_ivp(my_pde_rhs, ...)` with appropriate method and parameters.

## Dependencies

- **Required**: `numpy`, `scipy`, `joblib`
- **Acceleration**: `numba` >= 0.64.0 (JIT-compiled SOR kernel)
- **Plotting**: `matplotlib`, `scienceplots` (optional, needed for scripts)
- **Testing**: `pytest` (optional)

Python >= 3.10.
