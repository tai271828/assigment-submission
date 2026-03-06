"""
Monte Carlo DLA via random walkers.

Walkers are spawned one at a time at a random column in the top row and
perform a random walk on an N+1 x N+1 grid until they attach to the
cluster. Attachment occurs when a walker reaches a point adjacent to the
cluster, with probability sticking_probability. If a walker does not
attach within max_walker_steps steps, it is discarded and a new walker
is spawned. Walkers that step out of the top or bottom boundary are
respawned at a random column in the top row.

The main entry point is grow_dla_mc, which spawns n_walking_steps walkers
and returns the final state as a DLAMCResult. Intermediate states can be
captured via the post_growth callback.
"""

from ..core.result import DLAMCResult
from ..core.grid import get_neighbours

import numpy as np


def _make_mc_growth_step(
    growth_seed: tuple[int, int],
    N: int,
    sticking_probability: float,
    max_walker_steps: int = 100_000,
) -> callable:
    """
    Return a single-walker MC growth step function for DLA.

    Each call to the returned function spawns one walker at a random column
    in the top row, walks it until it attaches to the cluster or exhausts
    max_walker_steps steps, then returns the updated state.

    Args:
        growth_seed: (i, j) coordinates of the initial cluster point.
        N: Grid size. The grid has shape (N+1, N+1).
        sticking_probability: Probability that a walker attaches on contact,
            in [0, 1].
        max_walker_steps: Maximum number of steps before discarding a walker
            (default: 100,000).

    Returns:
        mc_growth_step: A function with signature
            mc_growth_step() -> (growth_mask, candidates)
    """
    # Set up initial conditions
    growth_mask = np.zeros(shape=(N + 1, N + 1), dtype=bool)
    growth_mask[*growth_seed] = True
    candidates = [c for c in get_neighbours(N, *growth_seed) if not growth_mask[*c]]

    def _get_random_column() -> int:
        """Return a random column index in [0, N)."""
        return np.random.randint(0, N)

    def _spawn_new_walker() -> tuple[int, int]:
        """Return coordinates for a new walker spawned at the top row."""
        return (_get_random_column(), N)

    def _random_step(i: int, j: int) -> tuple[int, int]:
        """
        Return new coordinates after one random step from (i, j).

        Steps into growth points are excluded. If no valid neighbours exist,
        the walker stays in place. If the selected neighbour is out of bounds
        (i.e. equals (i, j) due to clamping), the walker is respawned at a
        random column in the top row.
        """
        neighbours = np.array(get_neighbours(N, i, j))

        # Remove any neighbours that are in the growth object
        mask = ~growth_mask[neighbours[:, 0], neighbours[:, 1]]
        valid_neighbours = neighbours[mask]

        # Check there are valid neighbours
        if len(valid_neighbours) == 0:
            return (i, j)

        # Pick a random neighbour
        random_index = np.random.randint(0, len(valid_neighbours))
        new_coords = valid_neighbours[random_index]

        # Respawn walker if beyond top or bottom boundary
        if new_coords == (i, j):
            new_coords = _spawn_new_walker()

        return new_coords

    def mc_growth_step() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        """
        Spawn one walker and walk it until it attaches to the cluster.

        The walker is spawned at a random column in the top row and takes
        random steps, avoiding growth points. If it reaches a candidate point
        adjacent to the cluster, it attaches with probability
        sticking_probability. If it does not attach within max_walker_steps
        steps, it is discarded.

        Returns:
            growth_mask: Boolean array of shape (N+1, N+1), True at cluster points.
            candidates:  List of (i, j) coordinates adjacent to the cluster.
        """
        # Generate new walker
        walker_coords = _spawn_new_walker()

        for _ in range(max_walker_steps):
            # Check if walker attaches
            if (
                walker_coords in candidates
                and np.random.uniform(0, 1) < sticking_probability
            ):
                # Update growth mask
                growth_mask[walker_coords] = True

                # Update candidates
                candidates.remove(walker_coords)
                for new_candidate in get_neighbours(N, *walker_coords):
                    if not growth_mask[*new_candidate]:
                        candidates.append(new_candidate)

                # Break from loop
                break

            # Take a step
            walker_coords = _random_step(*walker_coords)

        return growth_mask, candidates

    return mc_growth_step


def grow_dla_mc(
    n_iter_growth,
    growth_seed: tuple[int, int],
    N: int,
    sticking_probability: float = 1.0,
    post_growth: callable = None,
    max_walker_steps=100_000,
) -> DLAMCResult:
    """
    Simulate Diffusion Limited Aggregation (DLA) using Monte Carlo random walkers.

    Spawns n_walking_steps walkers one at a time. Each walker performs a
    random walk until it attaches to the cluster or is discarded after
    max_walker_steps steps.

    Args:
        n_iter_growth: Number of growth steps to perform.
        growth_seed: (i, j) coordinates of the initial cluster point.
        N: Grid size. The grid has shape (N+1, N+1).
        sticking_probability: Probability that a walker attaches on contact,
            in [0, 1].
        post_growth: Optional callback f(step, growth_mask, candidates)
            called after each walker, e.g. to capture snapshots for animation.
        max_walker_steps: Maximum number of steps before discarding a walker
            (default: 100,000).

    Returns:
        DLAMCResult containing the final growth mask.
        Use post_growth to capture intermediate states.
    """
    mc_step = _make_mc_growth_step(
        growth_seed, N, sticking_probability, max_walker_steps
    )

    # Run simulation
    for k in range(n_iter_growth):
        growth_mask, candidates = mc_step()

        if post_growth is not None:
            post_growth(k + 1, growth_mask, candidates)

    return DLAMCResult(growth_mask)
