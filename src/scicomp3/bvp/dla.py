"""
This file contains implementations of the Diffusion Limited Aggregation (DLA)
- grow_dla_sor uses the SOR to implement DLA
- grow_dla_mc uses Monte Carlo to implement DLA
"""

from .solver import solve_bvp
from ..objects.growth import make_growth_step
from ..core.result import DLASORResult, DLAMCResult
from ..objects.walker import make_mc_growth_step

import numpy as np


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
    growth_step, growth_mask = make_growth_step(growth_seed, eta, N)
    for step in range(1, n_iter_growth + 1):
        result = solve_bvp(
            y0=y,
            method="sor",
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


def grow_dla_mc(
    n_walking_steps: int,
    growth_seed: tuple[int, int],
    N: int,
    sticking_probability: float = 1.0,
    post_growth: callable = None,
) -> DLAMCResult:
    """
    Simulate Diffusion Limited Aggregation (DLA) using Monte Carlo random walkers.

    Args:
        n_walking_steps: Number of steps to simulate the random walkers.
        growth_seed: (i, j) coordinates of the initial growth point.
        N: grid size
        sticking_probability: Probability that a walker sticks to the cluster on contact,
            in [0, 1]
        post_growth: Optional callback f(step, walker_mask, growth_mask, candidates)
            called after each growth step, e.g. to capture snapshots for animation.

    Returns:
        DLAMCResult: Only contains the final state.
            Use post_growth to capture intermediate states
    """
    mc_step = make_mc_growth_step(growth_seed, N, sticking_probability)

    # Generate random spawn columns for the walkers
    random_columns = np.random.randint(0, N + 1, n_walking_steps)

    # Run simulation
    for k, col in enumerate(random_columns):
        walker_mask, growth_mask, candidates = mc_step(col)

        if post_growth is not None:
            post_growth(k + 1, walker_mask, growth_mask, candidates)

    return DLAMCResult(walker_mask, growth_mask)
