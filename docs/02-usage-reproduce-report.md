# Assignment 3.1 — LBM Solver for the Karman Vortex Street

## What was implemented

A Lattice Boltzmann Method (LBM) solver for the Karman vortex street, integrated
into the `scicomp3` framework as the `"lb"` method in `solve_kvs()`.

The implementation lives in `src/scicomp3/kvs/methods.py` and follows the D2Q9
BGK algorithm from Gabor's lecture script (`lecture.10/lbm_karman-Drag_lift.py`).

### Algorithm (per timestep)

1. Compute macroscopic quantities (density, velocity) from distribution functions
2. BGK collision: relax f toward local equilibrium
3. Bounce-back: reflect populations at solid nodes (cylinder + channel walls)
4. Streaming: propagate distributions along lattice velocities
5. Boundary conditions: Zou-He inlet (fixed velocity), zero-gradient outlet

### Key design decisions

- **D2Q9 lattice constants** are defined at module level in `methods.py`
  (prefixed `_LB_*`) to be captured by numba's `@njit` for the equilibrium
  function.
- **Channel walls** (top/bottom) are combined with the cylinder mask into a
  `solid` mask for bounce-back, while drag/lift forces use only the cylinder
  mask.
- **Physical-to-lattice mapping**: the solver accepts a `U_lb` parameter
  (lattice inlet velocity) and derives `tau` from `Re`, `D_lb`, and `U_lb`.
  Output velocities are converted back to physical units.

## How to reproduce

### Prerequisites

```bash
cd <project-root>          # assignment-submission/
uv sync                    # install dependencies into .venv
```

### Run the LBM solver programmatically

```python
from scicomp3.core.config import KVSConfig
from scicomp3.kvs.solver import solve_kvs

# Fine grid needed for LBM stability at Re=150
config = KVSConfig(Re=150, Nx=440, Ny=82)

result = solve_kvs(
    config,
    method="lb",
    n_steps=10_000,
    plot_every=50,
    U_lb=0.1,          # lattice inlet velocity
)

# result.u          — final velocity field, shape (2, 441, 83)
# result.snapshots  — list of velocity snapshots for animation
# result.method     — "lb"
```

### Run the animation script

```bash
uv run python scripts/a3_1_kvs_lb_animation.py
```

This produces a GIF at `images/gifs/a3_1_kvs_lb.gif`.

### Compare with FD solver

```bash
uv run python scripts/a3_1_kvs_fd_animation.py
```

## Grid resolution and stability

The BGK collision operator requires `tau > 0.5`. Since
`tau = 3 * nu_lb + 0.5` and `nu_lb = U_lb * D_lb / Re`, stability
depends on the lattice diameter `D_lb = 2 * radius / dx`:

| Nx  | Ny | dx    | D_lb | tau (Re=150) | Stable? |
|-----|----|-------|------|--------------|---------|
| 220 | 41 | 0.010 | 10   | 0.520        | No      |
| 440 | 82 | 0.005 | 20   | 0.540        | Yes     |

**Rule of thumb**: keep `D_lb >= 20` for Re >= 100 with BGK.

## Known limitations (first prototype)

- **Density drift**: the zero-gradient outlet BC causes ~0.5% density increase
  per 1000 steps. Acceptable for 10k steps; for longer runs, consider a
  Zou-He outlet with prescribed density.
- **BGK only**: single-relaxation-time collision; TRT/MRT would improve
  stability at lower tau.
- **No pressure projection**: inherent to LBM — pressure is recovered from
  the equation of state `p = rho * cs^2`.
