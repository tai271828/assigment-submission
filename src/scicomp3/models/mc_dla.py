"""Monte Carlo Diffusion Limited Aggregation (MC-DLA) model.

Simulates DLA by releasing one random walker at a time from the top boundary
and letting it diffuse until it sticks to the growing cluster.

Grid convention (same as rest of scicomp3):
    - Array shape: (N+1, N+1)
    - axis 0 (rows)   → x-direction, periodic
    - axis 1 (cols)   → y-direction
    - j=0 is the bottom, j=N is the top (walker release boundary)

Random walker rules (Assignment 2.2.C, D):
    - Walker starts at a random x-position on the top row (j=N).
    - Each step moves one lattice site: up / down / left / right, chosen
      with equal probability 1/4.
    - Periodic boundary in x: wx = (wx ± 1) % (N+1).
    - If the walker leaves through the top (wy > N) or bottom (wy < 0)
      boundary, it is discarded and a new walker is released.
    - A walker is *not* allowed to step into a cluster site.
    - If the walker reaches a site adjacent to the cluster (N/E/S/W
      neighbour is a cluster site), it sticks with probability ``ps``.
      If it does not stick (ps < 1), the walk continues from the current
      position.

The η parameter is fixed at 1 in this model.  The sticking probability
``ps`` plays an analogous role: ps=1 gives standard DLA; ps<1 produces
more compact clusters.

References:
    Assignment Set 2, section 2.2 — Monte Carlo simulation of DLA
"""

import numpy as np


def _is_adjacent_to_cluster(wx, wy, cluster_mask, N):
    """Return True if (wx, wy) has at least one cluster neighbour."""
    neighbors = [
        ((wx + 1) % (N + 1), wy),
        ((wx - 1) % (N + 1), wy),
        (wx, wy + 1),
        (wx, wy - 1),
    ]
    return any(
        0 <= ny <= N and cluster_mask[nx, ny]
        for nx, ny in neighbors
    )


def run_mc_dla(
    N=100,
    n_steps=200,
    ps=1.0,
    rng=None,
    max_steps_per_walker=None,
):
    """Run a Monte Carlo DLA simulation.

    Args:
        N:                    Grid size; the domain has (N+1)×(N+1) points.
        n_steps:              Number of particles to add to the cluster.
        ps:                   Sticking probability in [0, 1].  ps=1 gives
                              standard DLA; smaller values produce more
                              compact clusters.
        rng:                  NumPy ``Generator``.  If None, a new
                              ``default_rng()`` is used.
        max_steps_per_walker: Maximum single-walker steps before giving up
                              (prevents infinite loops for very small grids
                              or unusual configurations).  Default: 10 * N².

    Returns:
        cluster_mask: Boolean array (N+1, N+1), True at cluster sites.
        walk_lengths: List of step counts for each walker that stuck.
    """
    if rng is None:
        rng = np.random.default_rng()
    if max_steps_per_walker is None:
        max_steps_per_walker = 10 * (N + 1) ** 2

    # Seed at bottom centre (same convention as PDE-DLA)
    cluster_mask = np.zeros((N + 1, N + 1), dtype=bool)
    cluster_mask[N // 2, 1] = True

    # Direction vectors: up, down, left, right
    dx = np.array([0, 0, -1, 1])
    dy = np.array([1, -1, 0, 0])

    walk_lengths = []
    added = 0

    while added < n_steps:
        # Release walker at random x on top boundary
        wx = int(rng.integers(0, N + 1))
        wy = N

        step_count = 0
        stuck = False

        while step_count < max_steps_per_walker:
            # Choose random direction
            d = int(rng.integers(0, 4))
            new_wx = (wx + dx[d]) % (N + 1)   # periodic in x
            new_wy = wy + dy[d]

            # Walker left through top or bottom → discard
            if new_wy > N or new_wy < 0:
                break   # start new walker

            # Walker cannot enter a cluster site → skip move
            if cluster_mask[new_wx, new_wy]:
                step_count += 1
                continue

            # Move the walker
            wx, wy = new_wx, new_wy
            step_count += 1

            # Check sticking condition
            if _is_adjacent_to_cluster(wx, wy, cluster_mask, N):
                if ps >= 1.0 or rng.random() < ps:
                    cluster_mask[wx, wy] = True
                    walk_lengths.append(step_count)
                    added += 1
                    stuck = True
                    break
                # else: ps < 1 and did not stick → continue walking

        if not stuck:
            # Walker escaped or hit max steps — release a new one
            pass

    return cluster_mask, walk_lengths
