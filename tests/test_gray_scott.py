"""Tests for the Gray-Scott reaction-diffusion system (Assignment 2.3)."""

import numpy as np
import pytest

from scicomp3.pde.gray_scott import (
    gray_scott_rhs,
    gray_scott_initial_conditions,
    gray_scott_stable_dt,
)
from scicomp3.ode.solver import solve_ivp


class TestGrayScottRHS:
    """Unit tests for the RHS function."""

    def test_output_shape_matches_input(self):
        N = 20
        state = np.random.default_rng(0).random((N, N, 2))
        d_state = gray_scott_rhs(0, state, Du=0.16, Dv=0.08, f=0.035, k=0.060)
        assert d_state.shape == state.shape

    def test_uniform_u1_v0_is_steady_state_without_diffusion(self):
        """u=1, v=0 is a trivial steady state: ∂u/∂t = f*(1-1) = 0, ∂v/∂t = -(f+k)*0 = 0."""
        N = 10
        state = np.zeros((N, N, 2))
        state[..., 0] = 1.0   # u = 1
        # v = 0 everywhere

        d_state = gray_scott_rhs(0, state, Du=0.16, Dv=0.08, f=0.035, k=0.060)
        np.testing.assert_allclose(d_state, 0.0, atol=1e-14)

    def test_du_dv_channel_independence(self):
        """With v=0 everywhere, ∂u/∂t = Dᵤ∇²u + f(1-u) only (no coupling to v)."""
        N = 10
        state = np.zeros((N, N, 2))
        state[..., 0] = 0.5  # u = 0.5, v = 0

        d_state = gray_scott_rhs(0, state, Du=0.16, Dv=0.08, f=0.035, k=0.060)

        # ∂u/∂t = Dᵤ·∇²u + f(1-u) = 0 (uniform u → ∇²u=0) + 0.035*(0.5) = 0.0175
        expected_dudt = 0.035 * (1.0 - 0.5)
        np.testing.assert_allclose(d_state[..., 0], expected_dudt, atol=1e-14)

        # ∂v/∂t = -(f+k)*v = 0 (since v=0)
        np.testing.assert_allclose(d_state[..., 1], 0.0, atol=1e-14)

    def test_dx_scaling(self):
        """Doubling dx multiplies the diffusion Laplacian contribution by 1/4.

        We isolate the pure-diffusion regime by setting v=0 everywhere
        (so uvv=0) and f=k=0 (no reaction/feed/kill terms).
        """
        N = 10
        rng = np.random.default_rng(7)
        u_field = rng.random((N, N))
        state = np.stack([u_field, np.zeros((N, N))], axis=-1)

        d1 = gray_scott_rhs(0, state, Du=1.0, Dv=1.0, f=0.0, k=0.0, dx=1.0)
        d2 = gray_scott_rhs(0, state, Du=1.0, Dv=1.0, f=0.0, k=0.0, dx=2.0)

        # With v=0, f=k=0: ∂u/∂t = Du·∇²u,  ∂v/∂t = Dv·∇²v = 0
        # Δ(dx=2) = Δ(dx=1) / 4
        np.testing.assert_allclose(d2, d1 / 4.0, atol=1e-14)


class TestGrayScottInitialConditions:
    def test_shape(self):
        state = gray_scott_initial_conditions(50)
        assert state.shape == (50, 50, 2)

    def test_values_in_unit_range(self):
        state = gray_scott_initial_conditions(50, rng=np.random.default_rng(0))
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0)

    def test_u_near_one_outside_square(self):
        """Outside the central square, u ≈ 1 (up to noise)."""
        N = 50
        noise = 0.05
        state = gray_scott_initial_conditions(N, noise_amplitude=noise,
                                              rng=np.random.default_rng(1))
        u = state[..., 0]
        # Corners of the domain are definitely outside the central square
        corners = [(0, 0), (0, N-1), (N-1, 0), (N-1, N-1)]
        for i, j in corners:
            assert abs(u[i, j] - 1.0) <= noise + 1e-9, \
                f"u at corner ({i},{j}) = {u[i,j]:.4f} should be close to 1"

    def test_v_nonzero_in_central_region(self):
        """v should be nonzero in the central square."""
        N = 50
        state = gray_scott_initial_conditions(N, noise_amplitude=0.0,
                                              rng=np.random.default_rng(2))
        v = state[..., 1]
        cx, cy = N // 2, N // 2
        assert v[cx, cy] > 0, "v should be nonzero in the central square"


class TestGrayScottStableDt:
    def test_default_parameters_are_stable(self):
        """Assignment default parameters (Du=0.16, Dv=0.08, dx=1) → dt=1 is stable."""
        dt_max = gray_scott_stable_dt(Du=0.16, Dv=0.08, dx=1.0, safety=1.0)
        assert dt_max >= 1.0, \
            f"Assignment dt=1.0 should be stable; max stable dt = {dt_max:.4f}"

    def test_larger_diffusivity_gives_smaller_dt(self):
        dt1 = gray_scott_stable_dt(Du=0.1, Dv=0.1, dx=1.0)
        dt2 = gray_scott_stable_dt(Du=0.5, Dv=0.5, dx=1.0)
        assert dt1 > dt2


class TestGrayScottIntegration:
    """Integration tests using solve_ivp."""

    def test_solve_produces_correct_shape(self):
        N = 10
        state0 = gray_scott_initial_conditions(N, rng=np.random.default_rng(0))
        result = solve_ivp(
            gray_scott_rhs,
            t_span=(0, 10),
            y0=state0,
            method="forward_euler",
            dt=1.0,
            args=(0.16, 0.08, 0.035, 0.060, 1.0),
            save_interval=5,
        )
        # result.y shape: (n_saved, N, N, 2)
        assert result.y.ndim == 4
        assert result.y.shape[1:] == (N, N, 2)

    def test_trivial_steady_state_unchanged(self):
        """u=1, v=0 is a steady state: the field should not change."""
        N = 10
        state0 = np.zeros((N, N, 2))
        state0[..., 0] = 1.0

        result = solve_ivp(
            gray_scott_rhs,
            t_span=(0, 50),
            y0=state0,
            method="forward_euler",
            dt=1.0,
            args=(0.16, 0.08, 0.035, 0.060, 1.0),
        )
        np.testing.assert_allclose(result.y[-1], state0, atol=1e-10)

    def test_v_grows_from_above_threshold_perturbation(self):
        """V grows when uv² > (f+k)v, i.e. v > (f+k)/u = 0.095.

        With u=1 and v=0.25 in a central region, the autocatalytic term
        uv²=0.0625 exceeds the kill term (f+k)v=0.02375, so V should grow
        initially from that region.
        """
        N = 20
        state0 = np.zeros((N, N, 2))
        state0[..., 0] = 1.0            # u = 1 everywhere
        cx = N // 2
        state0[cx-2:cx+2, cx-2:cx+2, 1] = 0.25   # v=0.25 in central square

        result = solve_ivp(
            gray_scott_rhs,
            t_span=(0, 50),
            y0=state0,
            method="forward_euler",
            dt=1.0,
            args=(0.16, 0.08, 0.035, 0.060, 1.0),
        )
        # V should spread / persist (not collapse to 0) in the early phase
        v_final_max = result.y[-1, :, :, 1].max()
        assert v_final_max > 0.01, \
            "V should not immediately collapse: autocatalytic term sustains it"
