"""Tests for PDE-based DLA (Assignment 2.1)."""

import numpy as np
import pytest

from scicomp3.models.dla import (
    find_growth_candidates,
    compute_growth_probabilities,
    run_dla,
)


class TestFindGrowthCandidates:
    def test_seed_at_j1_has_correct_candidates(self):
        """Seed at (5, 1) should yield candidates to its left, right, and above."""
        N = 10
        mask = np.zeros((N + 1, N + 1), dtype=bool)
        mask[5, 1] = True

        cands = find_growth_candidates(mask)

        # Neighbours in y (above): (5, 2)
        assert cands[5, 2], "Point above seed should be a candidate"
        # Neighbours in x (periodic): (4, 1) and (6, 1)
        assert cands[4, 1], "Left neighbour should be a candidate"
        assert cands[6, 1], "Right neighbour should be a candidate"
        # j=0 must be excluded
        assert not cands[5, 0], "Bottom row (j=0) must never be a candidate"
        # j=N must be excluded
        assert not cands[5, N], "Top row (j=N) must never be a candidate"
        # Seed itself must not be a candidate
        assert not cands[5, 1], "Seed itself must not be a candidate"

    def test_no_candidates_when_cluster_fills_interior(self):
        """A cluster filling all interior rows yields no candidates."""
        N = 4
        mask = np.ones((N + 1, N + 1), dtype=bool)
        cands = find_growth_candidates(mask)
        assert not np.any(cands)

    def test_periodic_x_boundary(self):
        """Seed at i=0 should yield a candidate at i=N (periodic)."""
        N = 10
        mask = np.zeros((N + 1, N + 1), dtype=bool)
        mask[0, 1] = True

        cands = find_growth_candidates(mask)
        assert cands[N, 1], "Periodic x: site at i=N should be a candidate"


class TestComputeGrowthProbabilities:
    def test_probabilities_sum_to_one(self):
        N = 10
        mask = np.zeros((N + 1, N + 1), dtype=bool)
        mask[5, 1] = True
        cands = find_growth_candidates(mask)

        concentration = np.random.default_rng(0).random((N + 1, N + 1))
        pg = compute_growth_probabilities(concentration, cands, eta=1.0)

        assert pytest.approx(pg.sum(), abs=1e-12) == 1.0

    def test_eta_zero_gives_uniform_probability(self):
        """η=0 → pg = c^0 = 1 for every candidate → uniform distribution."""
        N = 10
        mask = np.zeros((N + 1, N + 1), dtype=bool)
        mask[5, 1] = True
        cands = find_growth_candidates(mask)
        n_cands = cands.sum()

        concentration = np.random.default_rng(1).random((N + 1, N + 1))
        pg = compute_growth_probabilities(concentration, cands, eta=0.0)

        expected = 1.0 / n_cands
        np.testing.assert_allclose(pg, expected, atol=1e-12)

    def test_highest_concentration_candidate_has_highest_probability(self):
        """With η>0, the candidate with highest c should get the highest pg."""
        N = 10
        mask = np.zeros((N + 1, N + 1), dtype=bool)
        mask[5, 1] = True
        cands = find_growth_candidates(mask)

        concentration = np.zeros((N + 1, N + 1))
        # Set one candidate much higher
        candidate_positions = np.argwhere(cands)
        best = tuple(candidate_positions[0])
        concentration[best] = 10.0   # much higher than others

        pg = compute_growth_probabilities(concentration, cands, eta=1.0)
        assert pg.argmax() == 0, "Highest-concentration candidate should have highest pg"

    def test_fallback_uniform_when_all_concentrations_zero(self):
        N = 10
        mask = np.zeros((N + 1, N + 1), dtype=bool)
        mask[5, 1] = True
        cands = find_growth_candidates(mask)
        concentration = np.zeros((N + 1, N + 1))

        pg = compute_growth_probabilities(concentration, cands, eta=1.0)
        assert pytest.approx(pg.sum(), abs=1e-12) == 1.0
        np.testing.assert_allclose(pg, pg[0], atol=1e-12)


class TestRunDLA:
    @pytest.mark.slow
    def test_cluster_grows_by_n_steps(self):
        """Cluster should have seed + n_steps sites (if no early stop)."""
        N = 20
        n_steps = 10
        cluster_mask, _, n_iters = run_dla(
            N=N, n_steps=n_steps, eta=1.0, rng=np.random.default_rng(0)
        )
        # Seed = 1 site; after n_steps: 1 + n_steps sites
        assert cluster_mask.sum() == 1 + n_steps

    def test_n_iters_list_length_matches_growth_steps(self):
        N = 15
        n_steps = 5
        _, _, n_iters = run_dla(
            N=N, n_steps=n_steps, eta=1.0, rng=np.random.default_rng(1)
        )
        assert len(n_iters) == n_steps

    def test_cluster_connected_to_seed(self):
        """Every cluster site should be reachable from the seed."""
        N = 15
        n_steps = 8
        cluster_mask, _, _ = run_dla(
            N=N, n_steps=n_steps, eta=1.0, rng=np.random.default_rng(2)
        )

        # BFS from seed position (N//2, 1)
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

        # All cluster sites should be reachable
        assert np.all(visited[cluster_mask]), "Cluster must be connected"

    def test_boundary_rows_never_in_cluster(self):
        """Top (j=N) and bottom (j=0) boundary rows must never be grown into."""
        N = 15
        n_steps = 8
        cluster_mask, _, _ = run_dla(
            N=N, n_steps=n_steps, eta=1.0, rng=np.random.default_rng(3)
        )
        # Seed at j=1 is fine; j=0 should never be set by the growth loop
        # Note: seed is at [N//2, 1], not j=0
        assert not np.any(cluster_mask[:, 0]), "Bottom boundary row must not be grown"
        assert not np.any(cluster_mask[:, N]), "Top boundary row must not be grown"

    def test_concentration_boundary_conditions(self):
        """Final concentration must satisfy c(y=0)=0 and c(y=1)=1."""
        N = 15
        _, concentration, _ = run_dla(
            N=N, n_steps=5, eta=1.0, rng=np.random.default_rng(4)
        )
        np.testing.assert_allclose(concentration[:, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(concentration[:, N], 1.0, atol=1e-5)

    def test_eta_influences_growth(self):
        """Different η values should produce different cluster shapes."""
        N = 20
        n_steps = 15

        cluster_compact, _, _ = run_dla(
            N=N, n_steps=n_steps, eta=0.0, rng=np.random.default_rng(5)
        )
        cluster_default, _, _ = run_dla(
            N=N, n_steps=n_steps, eta=1.0, rng=np.random.default_rng(5)
        )
        cluster_branched, _, _ = run_dla(
            N=N, n_steps=n_steps, eta=3.0, rng=np.random.default_rng(5)
        )

        # All should grow the same number of sites
        assert cluster_compact.sum() == cluster_default.sum() == cluster_branched.sum()
        # They should differ in shape (not all identical)
        assert not np.array_equal(cluster_compact, cluster_default) or \
               not np.array_equal(cluster_default, cluster_branched), \
               "Different η values should produce different clusters"
