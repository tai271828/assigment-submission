"""
Diffusion-based DLA via SOR.

The aggregate grows one point at a time. At each step, the steady-state
diffusion equation is solved using SOR, and a candidate point adjacent to
the aggregate is selected with probability proportional to c^eta, where c
is the local concentration and eta controls the growth bias. Higher eta
produces more branching, needle-like structures; eta = 1 gives growth
probability directly proportional to concentration.

The grown aggregate is treated as a sink (concentration fixed at 0) in
each diffusion solve.

The main entry point is grow_dla_sor, which runs the full simulation and
returns the final state as a DLASORResult. Intermediate states can be
captured via the post_growth callback.
"""

from ..bvp.solver import solve_bvp
from ..core.result import DLASORResult
from ..core.grid import get_neighbours

import numpy as np


def _make_growth_step(
    growth_seed,
    eta: float = 1.8,
    N: int = 100,
) -> tuple[callable, np.ndarray]:
    """
    Returns a growth_step function.

    Args:
        growth_seed: coordinates of the initial grid point
        eta: parameter that determines the shape of object
        N: size of the grid

    Returns:
        - step: growth step function with signature
                step(y, **kwargs) -> growth_mask
                where y and growth_mask both have shape (N+1, N+1),
        - growth_mask: mask that signifies the initial configuration
            of growth points
    """
    # Set up initial conditions
    growth_mask = np.zeros(shape=(N + 1, N + 1), dtype=bool)
    growth_mask[*growth_seed] = True
    candidates = [c for c in get_neighbours(N, *growth_seed) if not growth_mask[*c]]

    # Define growth step
    def growth_step(y, **kwargs):
        # Compute probabilities
        concentrations = np.array([y[*c] for c in candidates])
        concentrations = np.clip(concentrations, 0, None)  # avoids negative values
        weights = concentrations**eta
        total = np.sum(weights)
        if total == 0:
            raise ValueError(
                "All candidate points have zero concentration. "
                "Check that boundary conditions produce a non-zero diffusion field,"
                "and that the SOR solver has converged."
            )
        p_values = weights / total

        # Draw a winner
        indices = np.arange(0, len(candidates))
        winning_index = np.random.choice(indices, p=p_values)
        winner = candidates[winning_index]

        # Update growth mask
        growth_mask[*winner] = True

        # Update candidates
        candidates.remove(winner)
        for new_candidate in get_neighbours(N, *winner):
            if not growth_mask[*new_candidate]:
                candidates.append(new_candidate)
        return growth_mask

    return growth_step, growth_mask


def grow_dla_sor(
    n_iter_growth,
    growth_seed,
    eta,
    y0,
    omega=1.9,
    tol=1e-5,
    max_iter_sor=100_000,
    post_step=None,
    post_growth=None,
    method="sor",
    **kwargs,
) -> DLASORResult:
    """
    Simulate Diffusion Limited Aggregation (DLA) using the diffusion equation.

    Alternates between solving the steady-state diffusion equation (via SOR)
    and growing the aggregate by one point, for n_iter_growth steps. The
    grown aggregate is treated as a sink object in the diffusion solve.

    Args:
        n_iter_growth: Number of growth steps to perform.
        growth_seed: (i, j) coordinates of the initial growth point.
        eta: Growth bias parameter. Higher values concentrate growth at
            points with high concentration.
        y0: Initial guess array of shape (N+1 x N+1).
        omega: SOR relaxation parameter, must be in [0, 2] (default: 1.9).
        tol: Convergence tolerance for the SOR solver (default: 1e-5).
        max_iter_sor: Maximum number of SOR iterations per growth step
            (default: 100,000).
        post_step: Optional callback f(k, y) -> y applied after each SOR
            iteration, e.g. to enforce boundary conditions.
        post_growth: Optional callback f(step, y, growth_mask) called after
            each growth step, e.g. to capture snapshots for animation.
        **kwargs: Additional arguments passed to solve_bvp.

    Returns:
        DLASORResult with y (solution array) and the growth_mask
    """
    y = y0.copy()
    N = len(y) - 1
    growth_step, growth_mask = _make_growth_step(growth_seed, eta, N)
    for step in range(1, n_iter_growth + 1):
        result = solve_bvp(
            y0=y,
            method=method,
            tol=tol,
            max_iter=max_iter_sor,
            post_step=post_step,
            sink_mask=growth_mask,
            omega=omega,
            **kwargs,
        )
        y = result.y

        try:
            growth_mask = growth_step(y)
        except ValueError:
            print(f"Growth stopped at step {step}: no viable candidates")
            break

        if post_growth is not None:
            post_growth(step, y, growth_mask)

    return DLASORResult(y, growth_mask)
