"""
[LEGACY]
Monte Carlo DLA via random walkers.

Walkers spawn at the top row and perform a random walk on an N+1 x N+1 grid.
When a walker reaches a point adjacent to the cluster, that point is added
to the cluster with probability sticking_probability. Walkers that step out
of the top or bottom boundary are respawned at a random column in the top row.

The main entry point is grow_dla_mc, which runs the full simulation and
returns the final state as a DLAMCResult. Intermediate states can be
captured via the post_growth callback.
"""

from ..core.result import DLAMCResult
from ..core.grid import get_neighbours

import numpy as np


def _random_step(N: int, i: int, j: int) -> tuple[int, int]:
    """
    Returns new coordinates after one random step from (i,j) on a N x N grid.
    If the step tries to move out of the top or bottom boundary,
    then the new coordinates are random respawn coordinates at the top boundary
    """
    neighbour_list = get_neighbours(N, i, j)
    random_index = np.random.randint(0, 4)
    new_coords = neighbour_list[random_index]
    if new_coords == (i, j):
        random_column = np.random.randint(1, N)
        new_coords = (random_column, N)
    return new_coords


def _make_mc_growth_step(
    growth_seed: tuple[int, int], N: int, sticking_probability: float
) -> callable:
    """
    Return a Monte Carlo growth step function for DLA.

    Walkers spawn at the top row and move randomly until they reach a
    candidate point adjacent to the cluster, at which point the cluster
    grows. State (walkers, growth mask, candidates) is maintained internally.

    Args:
        growth_seed: (i, j) coordinates of the initial cluster point.
        N: Grid size. The grid has shape (N+1, N+1).
        sticking_probability: Sticking probability

    Returns:
        mc_growth_step: A function with signature
            mc_growth_step(spawn_column) -> (walkers_mask, growth_mask, candidates)
    """
    # Set up initial conditions
    walkers_mask = np.zeros(shape=(N + 1, N + 1), dtype=int)
    growth_mask = np.zeros(shape=(N + 1, N + 1), dtype=bool)
    walkers_mask[*growth_seed] = 1
    growth_mask[*growth_seed] = True
    candidates = [c for c in get_neighbours(N, *growth_seed) if not growth_mask[*c]]

    def _grow_cluster():
        winners = []
        for candidate in candidates:
            if walkers_mask[candidate] > 0:
                winners.append(candidate)
        sticky_bools = np.random.uniform(0, 1, len(winners)) < sticking_probability
        for winner, winner_is_sticky in zip(winners, sticky_bools):
            if not winner_is_sticky:
                continue

            # Update growth mask
            growth_mask[winner] = True

            # Update candidates
            candidates.remove(winner)
            for new_candidate in get_neighbours(N, *winner):
                if not growth_mask[*new_candidate]:
                    candidates.append(new_candidate)

    def _move_walkers():
        new_walkers_mask = np.zeros_like(walkers_mask, dtype=int)
        occupied = np.argwhere((walkers_mask > 0) & ~growth_mask)
        for i, j in occupied:
            for _ in range(walkers_mask[i, j]):
                new_coords = _random_step(N, i, j)
                new_walkers_mask[new_coords] += 1
        walkers_mask[~growth_mask] = new_walkers_mask[~growth_mask]

    def mc_growth_step(
        spawn_column: int,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        """
        Perform one MC growth step.

        Spawns a walker at (spawn_column, N), grows the cluster where
        walkers overlap with candidates, then moves all walkers one step.

        Args:
            spawn_column: Column index at which to spawn the new walker.

        Returns:
            walkers_mask: Integer array of shape (N+1, N+1), walker counts per point.
            growth_mask:  Boolean array of shape (N+1, N+1), True at cluster points.
            candidates:   List of (i, j) coordinates adjacent to the cluster.
        """
        # Generate new walker
        walkers_mask[spawn_column, N] += 1

        # Grow cluster
        _grow_cluster()

        # Move walkers
        _move_walkers()

        return walkers_mask, growth_mask, candidates

    return mc_growth_step


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
    mc_step = _make_mc_growth_step(growth_seed, N, sticking_probability)

    # Generate random spawn columns for the walkers
    random_columns = np.random.randint(0, N + 1, n_walking_steps)

    # Run simulation
    for k, col in enumerate(random_columns):
        walker_mask, growth_mask, candidates = mc_step(col)

        if post_growth is not None:
            post_growth(k + 1, walker_mask, growth_mask, candidates)

    return DLAMCResult(walker_mask, growth_mask)
