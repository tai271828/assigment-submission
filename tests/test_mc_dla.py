"""Tests for Monte Carlo DLA (Assignment 2.2)."""

import numpy as np
import pytest

from scicomp3.models.mc_dla import run_mc_dla, _is_adjacent_to_cluster


class TestIsAdjacentToCluster:
    def test_direct_neighbor_is_adjacent(self):
        N = 10
        mask = np.zeros((N + 1, N + 1), dtype=bool)
        mask[5, 3] = True  # cluster site

        assert _is_adjacent_to_cluster(5, 4, mask, N)   # above
        assert _is_adjacent_to_cluster(5, 2, mask, N)   # below
        assert _is_adjacent_to_cluster(4, 3, mask, N)   # left
        assert _is_adjacent_to_cluster(6, 3, mask, N)   # right

    def test_non_neighbor_not_adjacent(self):
        N = 10
        mask = np.zeros((N + 1, N + 1), dtype=bool)
        mask[5, 3] = True

        assert not _is_adjacent_to_cluster(3, 3, mask, N)  # two steps away
        assert not _is_adjacent_to_cluster(5, 5, mask, N)  # two steps above

    def test_periodic_x_adjacency(self):
        """Seed at i=0, j=3: the site at i=N, j=3 should be adjacent."""
        N = 10
        mask = np.zeros((N + 1, N + 1), dtype=bool)
        mask[0, 3] = True

        assert _is_adjacent_to_cluster(N, 3, mask, N)


class TestRunMcDla:
    def test_cluster_grows_to_n_steps(self):
        N = 15
        n_steps = 5
        cluster_mask, walk_lengths = run_mc_dla(
            N=N, n_steps=n_steps, ps=1.0, rng=np.random.default_rng(0)
        )
        # seed + n_steps new sites
        assert cluster_mask.sum() == 1 + n_steps
        assert len(walk_lengths) == n_steps

    def test_walk_lengths_are_positive(self):
        N = 15
        n_steps = 5
        _, walk_lengths = run_mc_dla(
            N=N, n_steps=n_steps, ps=1.0, rng=np.random.default_rng(1)
        )
        assert all(w > 0 for w in walk_lengths)

    def test_cluster_sites_never_on_boundaries(self):
        """j=0 and j=N must never be grown into (they are the BC rows)."""
        N = 20
        n_steps = 10
        cluster_mask, _ = run_mc_dla(
            N=N, n_steps=n_steps, ps=1.0, rng=np.random.default_rng(2)
        )
        assert not np.any(cluster_mask[:, 0])
        assert not np.any(cluster_mask[:, N])

    def test_cluster_connected(self):
        """Every cluster site must be connected to the seed via BFS."""
        N = 15
        n_steps = 8
        cluster_mask, _ = run_mc_dla(
            N=N, n_steps=n_steps, ps=1.0, rng=np.random.default_rng(3)
        )

        visited = np.zeros_like(cluster_mask)
        seed = (N // 2, 1)
        queue = [seed]
        visited[seed] = True

        while queue:
            ci, cj = queue.pop()
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = (ci + di) % (N + 1), cj + dj
                if 0 <= nj <= N and cluster_mask[ni, nj] and not visited[ni, nj]:
                    visited[ni, nj] = True
                    queue.append((ni, nj))

        assert np.all(visited[cluster_mask]), "MC DLA cluster must be connected"

    def test_ps_less_than_one_produces_longer_walks(self):
        """Lower ps → walker bounces near cluster before sticking → longer walks.

        This property holds statistically; we use a large enough sample on a
        grid where the cluster is still growing upward (not space-filling).
        """
        # Use a very small ps (0.01) to make the effect unambiguous:
        # on average each walker must attempt sticking ~100 times.
        N = 30
        n_steps = 30

        _, walks_ps1 = run_mc_dla(N=N, n_steps=n_steps, ps=1.0,
                                   rng=np.random.default_rng(77))
        _, walks_ps001 = run_mc_dla(N=N, n_steps=n_steps, ps=0.01,
                                     rng=np.random.default_rng(77))

        assert np.mean(walks_ps001) > np.mean(walks_ps1), \
            "ps=0.01 should produce longer walks than ps=1.0 on average"

    def test_ps_one_and_ps_below_one_differ_in_morphology(self):
        """ps=1.0 and ps=0.1 should produce different clusters."""
        N = 20
        n_steps = 15

        mask_ps1, _ = run_mc_dla(N=N, n_steps=n_steps, ps=1.0,
                                  rng=np.random.default_rng(5))
        mask_ps01, _ = run_mc_dla(N=N, n_steps=n_steps, ps=0.1,
                                   rng=np.random.default_rng(5))

        # Both same size, different shapes
        assert mask_ps1.sum() == mask_ps01.sum()
        assert not np.array_equal(mask_ps1, mask_ps01), \
            "Different ps values should produce different cluster shapes"
