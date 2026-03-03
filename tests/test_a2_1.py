"""Tests for DLA eta sweep via SOR (Assignment 2.1).

Exercises the same code path as scripts/a2_1_dla_eta_sweep.py with
smaller grid / fewer steps so tests finish quickly in CI.

Parameters lifted from the script:
    N=50, N_STEPS=100, SEED=42, TOL=1e-4, MAX_ITER=2000,
    ETAS=[0.5, 1.0, 2.0, 4.0]

Reduced here to N=30, N_STEPS=20 for speed.
"""

from pathlib import Path

import numpy as np
import pytest

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.dla import grow_dla_sor
from scicomp3.bvp.omega import get_optimal_omega

# -- Reduced parameters for CI speed ------------------------------------------
N = 30
N_STEPS = 20
SEED = 42
OMEGA = get_optimal_omega(N)
TOL = 1e-4
MAX_ITER = 2_000
ETAS = [0.5, 1.0, 2.0, 4.0]
GROWTH_SEED = (N // 2, N // 2)


def fixed_bc(k, y):
    """Enforce diffusion BCs after each SOR iteration."""
    apply_diffusion_bc(y)
    return y


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def grid():
    return Grid2D(N=N, L=1.0)


@pytest.fixture(params=ETAS, ids=[f"eta_{e}" for e in ETAS])
def dla_result(request, grid):
    """Run DLA for a single eta value and return (eta, result)."""
    eta = request.param
    np.random.seed(SEED)
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)
    result = grow_dla_sor(
        N_STEPS,
        GROWTH_SEED,
        eta,
        c0,
        OMEGA,
        TOL,
        max_iter_sor=MAX_ITER,
        post_step=fixed_bc,
    )
    return eta, result


@pytest.fixture
def growth_order(grid):
    """Run DLA with post_growth tracking, mirroring the script's logic."""
    eta = 1.0
    np.random.seed(SEED)
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)

    order = np.full((N + 1, N + 1), np.nan)
    order[GROWTH_SEED] = 0

    def track(step, y, growth_mask):
        new = growth_mask & np.isnan(order)
        order[new] = step

    grow_dla_sor(
        N_STEPS,
        GROWTH_SEED,
        eta,
        c0,
        OMEGA,
        TOL,
        max_iter_sor=MAX_ITER,
        post_step=fixed_bc,
        post_growth=track,
    )
    return order


# -- Tests: per-eta properties ------------------------------------------------


class TestDLAEtaSweep:
    """Tests for DLA growth across eta values (Assignment 2.1)."""

    def test_result_shapes(self, dla_result, grid):
        """y and growth_mask have shape (N+1, N+1)."""
        _, result = dla_result
        assert result.y.shape == grid.shape
        assert result.growth_mask.shape == grid.shape

    def test_seed_in_cluster(self, dla_result):
        """The seed site must always be part of the cluster."""
        _, result = dla_result
        assert result.growth_mask[GROWTH_SEED], "Seed not in growth mask"

    def test_cluster_grows(self, dla_result):
        """Cluster should contain more than just the seed."""
        eta, result = dla_result
        n_sites = np.count_nonzero(result.growth_mask)
        assert n_sites > 1, f"eta={eta}: cluster did not grow beyond seed"

    def test_boundary_bottom(self, dla_result):
        """Bottom boundary c=0 is preserved."""
        eta, result = dla_result
        assert np.allclose(result.y[:, 0], 0), f"eta={eta}: bottom BC violated"

    def test_boundary_top(self, dla_result):
        """Top boundary c=1 is preserved."""
        eta, result = dla_result
        assert np.allclose(result.y[:, -1], 1), f"eta={eta}: top BC violated"

    def test_concentration_bounded(self, dla_result):
        """Concentration should stay within [0, 1]."""
        eta, result = dla_result
        assert np.all(result.y >= -1e-10), f"eta={eta}: negative concentration"
        assert np.all(result.y <= 1 + 1e-10), f"eta={eta}: concentration > 1"

    def test_sink_sites_zero(self, dla_result):
        """Cluster sites act as sinks — concentration should be zero there.

        The last-grown site may retain its pre-growth concentration because
        grow_dla_sor returns the mask *after* the final growth step but the
        SOR solve ran *before* it.  So we allow at most one non-zero site.
        """
        eta, result = dla_result
        sink_vals = result.y[result.growth_mask]
        n_nonzero = np.count_nonzero(np.abs(sink_vals) > 1e-8)
        assert (
            n_nonzero <= 1
        ), f"eta={eta}: {n_nonzero} sink sites non-zero (expected at most 1)"


# -- Tests: growth-order tracking ---------------------------------------------


class TestGrowthOrder:
    """Tests for the post_growth callback used in the script."""

    def test_seed_is_step_zero(self, growth_order):
        """Seed site should have growth order 0."""
        assert growth_order[GROWTH_SEED] == 0

    def test_order_values_sequential(self, growth_order):
        """Growth steps should be positive integers (except seed at 0)."""
        valid = growth_order[~np.isnan(growth_order)]
        assert np.all(valid >= 0), "Negative growth order found"
        assert np.all(valid == np.floor(valid)), "Non-integer growth order"

    def test_cluster_size_matches_order(self, growth_order):
        """Number of non-NaN entries should exceed 1 (seed + grown sites)."""
        n_sites = np.count_nonzero(~np.isnan(growth_order))
        assert n_sites > 1, "Growth order records only the seed"


# -- Tests: reproducibility and cross-eta properties --------------------------


class TestReproducibility:
    """Fixed seed must produce identical clusters."""

    def test_identical_runs(self, grid):
        """Two runs with the same seed + eta yield the same result."""
        eta = 1.0
        masks = []
        fields = []
        for _ in range(2):
            np.random.seed(SEED)
            c0 = np.zeros(grid.shape)
            apply_diffusion_bc(c0)
            r = grow_dla_sor(
                N_STEPS,
                GROWTH_SEED,
                eta,
                c0,
                OMEGA,
                TOL,
                max_iter_sor=MAX_ITER,
                post_step=fixed_bc,
            )
            masks.append(r.growth_mask)
            fields.append(r.y)
        assert np.array_equal(masks[0], masks[1]), "Masks differ across runs"
        assert np.allclose(fields[0], fields[1]), "Fields differ across runs"


# -- Tests: regression against stored reference --------------------------------

# Parameters matching scripts/a2_1_dla_by_sor.py exactly
_REF_N = 50
_REF_N_STEPS = 100
_REF_ETA = 1.0
_REF_SEED = 42
_REF_OMEGA = get_optimal_omega(_REF_N)
_REF_TOL = 1e-4
_REF_MAX_ITER = 2_000
_REF_GROWTH_SEED = (_REF_N // 2, _REF_N // 2)

_DATA_DIR = Path(__file__).parent.parent / "data"
_REF_FILE = _DATA_DIR / "dla_reference_eta1.0.npz"


@pytest.fixture(scope="module")
def reference_data():
    """Load stored reference arrays from the known-good run."""
    assert _REF_FILE.exists(), f"Reference file not found: {_REF_FILE}"
    return np.load(_REF_FILE)


@pytest.fixture(scope="module")
def regression_result():
    """Run DLA with the same parameters as scripts/a2_1_dla_by_sor.py."""
    np.random.seed(_REF_SEED)
    c0 = np.zeros((_REF_N + 1, _REF_N + 1))
    apply_diffusion_bc(c0)

    growth_order = np.full((_REF_N + 1, _REF_N + 1), np.nan)
    growth_order[_REF_GROWTH_SEED] = 0

    def track(step, y, growth_mask):
        new = growth_mask & np.isnan(growth_order)
        growth_order[new] = step

    result = grow_dla_sor(
        _REF_N_STEPS,
        _REF_GROWTH_SEED,
        _REF_ETA,
        c0,
        _REF_OMEGA,
        _REF_TOL,
        max_iter_sor=_REF_MAX_ITER,
        post_step=fixed_bc,
        post_growth=track,
    )
    return result, growth_order


class TestDLAEtaSweepSameContent:
    """Regression tests: grow_dla_sor must reproduce the stored reference.

    Reference file: data/dla_reference_eta1.0.npz
    Generated with: N=50, N_STEPS=100, ETA=1.0, SEED=42,
                    OMEGA=get_optimal_omega(50), TOL=1e-4, MAX_ITER=2000,
                    growth_seed=(25, 25).
    """

    def test_growth_mask_identical(self, regression_result, reference_data):
        """The grown cluster must be bit-identical to the reference."""
        result, _ = regression_result
        np.testing.assert_array_equal(
            result.growth_mask,
            reference_data["growth_mask"],
            err_msg="growth_mask differs from reference",
        )

    def test_concentration_field_close(self, regression_result, reference_data):
        """Concentration field must match reference within SOR tolerance."""
        result, _ = regression_result
        np.testing.assert_allclose(
            result.y,
            reference_data["y"],
            atol=1e-10,
            rtol=1e-10,
            err_msg="Concentration field differs from reference",
        )

    def test_growth_order_identical(self, regression_result, reference_data):
        """Growth order (which site was added at which step) must match."""
        _, growth_order = regression_result
        ref_order = reference_data["growth_order"]
        # Compare only non-NaN entries (NaN != NaN by IEEE)
        mask = ~np.isnan(ref_order)
        assert np.array_equal(
            mask, ~np.isnan(growth_order)
        ), "Different sites have growth-order entries"
        np.testing.assert_array_equal(
            growth_order[mask],
            ref_order[mask],
            err_msg="Growth order values differ from reference",
        )

    def test_cluster_size(self, regression_result, reference_data):
        """Cluster size must match reference exactly."""
        result, _ = regression_result
        expected = np.count_nonzero(reference_data["growth_mask"])
        actual = np.count_nonzero(result.growth_mask)
        assert actual == expected, f"Cluster size {actual} != reference {expected}"

    def test_concentration_stats(self, regression_result, reference_data):
        """Summary statistics of the concentration field must match."""
        result, _ = regression_result
        ref_y = reference_data["y"]
        np.testing.assert_allclose(result.y.mean(), ref_y.mean(), rtol=1e-10)
        np.testing.assert_allclose(result.y.std(), ref_y.std(), rtol=1e-10)
