"""
This file contains the implementation of the Diffusion Limited Aggregation
obtained via the diffusion equation
"""

from .solver import solve_bvp
from ..objects.growth import make_growth_step
from ..core.result import DLASORResult

def grow_dla_sor(n_iter_growth,
                 growth_seed,
                 eta,
                 y0,
                 omega=1.9,
                 tol=1e-5,
                 max_iter_sor=100_000,
                 post_step=None,
                 **kwargs) -> DLASORResult:
    """
    Simulate Diffusion Limited Aggregation (DLA) using the diffusion equation.

    Alternates between solving the steady-state diffusion equation (via SOR)
    and growing the aggregate by one point, for n_iter_growth steps. The
    grown aggregate is treated as an insulating object in the diffusion solve.

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
        **kwargs: Additional arguments passed to solve_bvp.

    Returns:
        DLASORResult with y (solution array) and the growth_mask
    """
    y = y0.copy()
    N = len(y) - 1
    growth_step, growth_mask = make_growth_step(growth_seed, eta, N)
    for _ in range(n_iter_growth):
        result = solve_bvp(
            y0=y,
            method="sor",
            tol=tol,
            max_iter=max_iter_sor,
            post_step=post_step,
            insulator_mask=growth_mask,
            omega=omega,
            **kwargs
        )
        y = result.y

        growth_mask = growth_step(y)
    return DLASORResult(y, growth_mask)

