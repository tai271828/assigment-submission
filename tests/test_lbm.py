"""Tests for the Lattice Boltzmann Method solver."""

import numpy as np
import pytest

from scicomp3.lbm import D2Q9, LBMSolver, cylinder_mask, rectangle_mask
from scicomp3.lbm.solver import tau_from_re, compute_re


class TestD2Q9:
    """Test D2Q9 lattice model constants and equilibrium."""

    def test_weights_sum_to_one(self):
        assert np.isclose(np.sum(D2Q9.w), 1.0)

    def test_velocities_shape(self):
        assert D2Q9.e.shape == (9, 2)

    def test_opposite_is_involution(self):
        """opposite[opposite[i]] == i for all i."""
        opp = D2Q9.opposite
        for i in range(9):
            assert opp[opp[i]] == i

    def test_opposite_reverses_velocity(self):
        """e[opposite[i]] == -e[i] for all i."""
        for i in range(9):
            j = D2Q9.opposite[i]
            np.testing.assert_array_equal(D2Q9.e[j], -D2Q9.e[i])

    def test_equilibrium_at_rest(self):
        """At rest (u=0), f_eq = w_i * rho."""
        Nx, Ny = 10, 10
        rho = 2.0 * np.ones((Nx, Ny))
        ux = np.zeros((Nx, Ny))
        uy = np.zeros((Nx, Ny))

        feq = D2Q9.equilibrium(rho, ux, uy)
        for i in range(9):
            expected = D2Q9.w[i] * rho
            np.testing.assert_allclose(feq[i], expected, rtol=1e-12)

    def test_equilibrium_mass_conservation(self):
        """Sum of f_eq over directions equals rho."""
        Nx, Ny = 10, 10
        rho = np.random.uniform(0.9, 1.1, (Nx, Ny))
        ux = np.random.uniform(-0.05, 0.05, (Nx, Ny))
        uy = np.random.uniform(-0.05, 0.05, (Nx, Ny))

        feq = D2Q9.equilibrium(rho, ux, uy)
        rho_check = np.sum(feq, axis=0)
        np.testing.assert_allclose(rho_check, rho, rtol=1e-12)

    def test_equilibrium_momentum_conservation(self):
        """Sum of e_i * f_eq_i equals rho * u."""
        Nx, Ny = 10, 10
        rho = np.random.uniform(0.9, 1.1, (Nx, Ny))
        ux = np.random.uniform(-0.05, 0.05, (Nx, Ny))
        uy = np.random.uniform(-0.05, 0.05, (Nx, Ny))

        feq = D2Q9.equilibrium(rho, ux, uy)

        mx = np.zeros((Nx, Ny))
        my = np.zeros((Nx, Ny))
        for i in range(9):
            mx += D2Q9.e[i, 0] * feq[i]
            my += D2Q9.e[i, 1] * feq[i]

        np.testing.assert_allclose(mx, rho * ux, rtol=1e-12)
        np.testing.assert_allclose(my, rho * uy, rtol=1e-12)


class TestObstacles:
    """Test obstacle mask generation."""

    def test_cylinder_mask_center(self):
        mask = cylinder_mask(50, 50, 25, 25, 5)
        assert mask[25, 25] == True  # center is inside

    def test_cylinder_mask_outside(self):
        mask = cylinder_mask(50, 50, 25, 25, 5)
        assert mask[0, 0] == False  # corner is outside

    def test_cylinder_mask_shape(self):
        mask = cylinder_mask(100, 50, 25, 25, 5)
        assert mask.shape == (100, 50)

    def test_rectangle_mask(self):
        mask = rectangle_mask(50, 50, 10, 20, 10, 20)
        assert mask[15, 15] == True
        assert mask[0, 0] == False
        assert mask.shape == (50, 50)


class TestLBMSolver:
    """Test LBM solver basic functionality."""

    def test_initialization(self):
        Nx, Ny = 50, 20
        obstacle = np.zeros((Nx, Ny), dtype=bool)
        solver = LBMSolver(Nx, Ny, tau=0.8, obstacle=obstacle, u_inlet=0.04)

        assert solver.f.shape == (9, Nx, Ny)
        assert solver.rho.shape == (Nx, Ny)
        np.testing.assert_allclose(solver.rho, 1.0, rtol=1e-10)

    def test_viscosity_from_tau(self):
        """nu = cs² * (tau - 0.5) = (tau - 0.5)/3."""
        obstacle = np.zeros((10, 10), dtype=bool)
        solver = LBMSolver(10, 10, tau=1.0, obstacle=obstacle)
        np.testing.assert_allclose(solver.nu, 1.0 / 6.0)

        solver2 = LBMSolver(10, 10, tau=0.8, obstacle=obstacle)
        np.testing.assert_allclose(solver2.nu, 0.3 / 3.0)

    def test_single_step_no_crash(self):
        Nx, Ny = 30, 15
        obstacle = cylinder_mask(Nx, Ny, 7, 7, 3)
        solver = LBMSolver(Nx, Ny, tau=0.8, obstacle=obstacle, u_inlet=0.04)
        solver.step()

        assert not np.any(np.isnan(solver.rho))
        assert not np.any(np.isinf(solver.rho))

    def test_solve_returns_result(self):
        Nx, Ny = 30, 15
        obstacle = cylinder_mask(Nx, Ny, 7, 7, 3)
        solver = LBMSolver(Nx, Ny, tau=0.8, obstacle=obstacle, u_inlet=0.04)
        result = solver.solve(10, save_every=5)

        assert result.rho.shape == (Nx, Ny)
        assert result.ux.shape == (Nx, Ny)
        assert result.uy.shape == (Nx, Ny)
        assert len(result.history) == 2  # saved at step 5 and 10

    def test_obstacle_velocity_zero(self):
        """Velocity should be zero inside the obstacle."""
        Nx, Ny = 30, 15
        obstacle = cylinder_mask(Nx, Ny, 7, 7, 3)
        solver = LBMSolver(Nx, Ny, tau=0.8, obstacle=obstacle, u_inlet=0.04)
        solver.solve(50)

        np.testing.assert_array_equal(solver.ux[obstacle], 0.0)
        np.testing.assert_array_equal(solver.uy[obstacle], 0.0)

    @pytest.mark.slow
    def test_poiseuille_flow(self):
        """Test that channel flow without obstacle converges to parabolic profile.

        Validates:
        1. The normalized velocity profile is parabolic (correct shape).
        2. Profile is symmetric about channel center.
        3. Velocity is approximately zero at walls (no-slip).
        """
        Nx, Ny = 100, 21
        obstacle = np.zeros((Nx, Ny), dtype=bool)
        u_inlet = 0.04
        tau = 0.8
        solver = LBMSolver(Nx, Ny, tau=tau, obstacle=obstacle, u_inlet=u_inlet)
        solver.solve(8000)

        H = Ny - 1
        y = np.arange(Ny)
        parabola = y * (H - y) / (H / 2.0) ** 2  # normalized parabola, peak=1

        # 1) Shape: normalized profile at mid-channel should be parabolic
        mid = Nx // 2
        ux_mid = solver.ux[mid, :]
        peak = np.max(ux_mid)
        assert peak > 0, "Flow must be driven by inlet"
        ux_norm = ux_mid / peak
        np.testing.assert_allclose(ux_norm[2:-2], parabola[2:-2], atol=0.05)

        # 2) Symmetry: profile should be symmetric about center
        np.testing.assert_allclose(
            ux_mid[:Ny // 2], ux_mid[Ny - 1:Ny // 2:-1], rtol=0.01
        )

        # 3) No-slip: velocity should be small at walls
        assert np.abs(solver.ux[mid, 0]) < 1e-3
        assert np.abs(solver.ux[mid, -1]) < 1e-3


class TestUnitConversions:
    """Test Re/tau conversion functions."""

    def test_tau_from_re(self):
        tau = tau_from_re(Re=100, u_inlet=0.04, D_lb=20)
        nu = 0.04 * 20 / 100  # = 0.008
        expected_tau = 3.0 * nu + 0.5  # = 0.524
        np.testing.assert_allclose(tau, expected_tau)

    def test_compute_re(self):
        re = compute_re(u_inlet=0.04, D_lb=20, nu_lb=0.008)
        np.testing.assert_allclose(re, 100.0)

    def test_roundtrip(self):
        """tau_from_re and compute_re should be inverses."""
        Re_target = 150
        u_lb = 0.05
        D_lb = 30

        tau = tau_from_re(Re_target, u_lb, D_lb)
        nu_lb = (tau - 0.5) / 3.0
        Re_computed = compute_re(u_lb, D_lb, nu_lb)
        np.testing.assert_allclose(Re_computed, Re_target)
