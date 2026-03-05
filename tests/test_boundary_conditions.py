"""Test boundary condition enforcement for wave equation solver.

Validates that boundary conditions Ψ(0,t)=0 and Ψ(L,t)=0 are maintained
across all time steps for all test cases.

This is a pytest-compatible version of scripts/a1_1_validation_boundary_conditions.py
"""

import numpy as np
import pytest

from scicomp3 import Grid1D, solve_ivp, wave1d_rhs
from scicomp3.core.result import ODEResult, find_y
from scicomp3.pde.wave import (
    initial_condition_case_i,
    initial_condition_case_ii,
    initial_condition_case_iii,
)


def fixed_ends(t, y):
    """Enforce fixed boundary conditions: psi=0 at both ends."""
    y[0, 0] = 0
    y[-1, 0] = 0
    return y


# Parameters (same as original script)
c = 1
L = 1
N = 90
dt = 1e-3
T_sim = 1.0  # shorter than full sim for faster tests

# Setup grid
grid = Grid1D(N=N - 1, L=L)

# Define test cases (same as original script)
test_cases = [
    ("Case i", initial_condition_case_i),
    ("Case ii", initial_condition_case_ii),
    ("Case iii", initial_condition_case_iii),
]


def is_zero_at_the_x_ends(result: ODEResult, t):
    """
    Checks whether the result evaluates at zero for the given t at the x-boundaries"""
    simulated_array = find_y(result, t)
    zero_at_x_start = simulated_array[0][0] == 0
    zero_at_x_end = simulated_array[-1][0] == 0

    return zero_at_x_start and zero_at_x_end


def validate_boundary_conditions(
    result: ODEResult, validate_function=is_zero_at_the_x_ends, verbose=False
):
    """
    Validates the boundary conditions on the full t domain.
    The validate_function should be of the form func(result, t)
    """
    for t in result.t:
        assert validate_function(
            result, t
        ), f"The function does not evaluate to zero at t={t} at (one of) the x-boundaries"
    if verbose:
        print("The boundary conditions were succesfully validated")


def run_simulation(ic_func):
    """Run wave equation simulation with given initial condition."""
    psi0 = ic_func(grid.x)
    # initialized the fixed point required by the boundary conditions
    psi0[0] = 0
    psi0[-1] = 0

    v0 = np.zeros(N)
    y0 = np.column_stack([psi0, v0])

    result = solve_ivp(
        wave1d_rhs,
        t_span=(0, T_sim),
        y0=y0,
        method="symplectic_euler",
        dt=dt,
        args=(c, L, N),
        post_step=fixed_ends,
    )
    return result


# --- Tests ---


@pytest.mark.parametrize("name,ic_func", test_cases)
def test_boundary_conditions(name, ic_func):
    """Validate boundary conditions are satisfied at all time steps.

    This is the pytest version of the original for-loop validation.
    """
    print(f"\nProcessing {name}...")

    result = run_simulation(ic_func)
    validate_boundary_conditions(result, verbose=True)


class TestBoundaryConditionsWithoutPostStep:
    """Tests demonstrating the need for post_step (regression tests)."""

    def test_without_post_step_boundaries_may_drift(self):
        """Without post_step, boundary conditions may not be maintained.

        This test documents the expected behavior - np.roll causes boundary
        pollution, so post_step is necessary to maintain exact zeros.
        """
        psi0 = initial_condition_case_ii(grid.x)
        psi0[0] = 0
        psi0[-1] = 0
        v0 = np.zeros(N)
        y0 = np.column_stack([psi0, v0])

        # Run WITHOUT post_step
        result = solve_ivp(
            wave1d_rhs,
            t_span=(0, T_sim),
            y0=y0,
            method="symplectic_euler",
            dt=dt,
            args=(c, L, N),
            post_step=None,  # No boundary enforcement!
        )

        amplitudes = result.y[:, :, 0]
        left_max = np.max(np.abs(amplitudes[:, 0]))
        right_max = np.max(np.abs(amplitudes[:, -1]))

        # Document: without post_step, boundaries may be nonzero
        # This is expected due to np.roll boundary pollution
        # The test passes regardless - it's documenting behavior
        print(f"Without post_step: left_max={left_max}, right_max={right_max}")
