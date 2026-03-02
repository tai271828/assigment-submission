"""
Growth object for Diffusion Limited Aggregation (DLA).

The growth state (current aggregate and candidate points) is maintained
internally by the closure returned by make_growth_step.
"""

import numpy as np


def _get_neighbours(N, i, j):
    """
    Helper function.
    Returns coordinates of the four neighbours of the given point (i,j)
    for a grid with size N.

    Wraps around for the j coordinate, and clamps for the i coordinate
    """
    i_min = (j - 1) % (N + 1)
    i_plus = (j + 1) % (N + 1)
    j_min = max(i - 1, 0)
    j_plus = min(i + 1, N)
    return [(i_min, j), (i_plus, j), (i, j_min), (i, j_plus)]


def make_growth_step(
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
    candidates = [c for c in _get_neighbours(N, *growth_seed) if not growth_mask[*c]]

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
        for new_candidate in _get_neighbours(N, *winner):
            if not growth_mask[*new_candidate]:
                candidates.append(new_candidate)
        return growth_mask

    return growth_step, growth_mask
