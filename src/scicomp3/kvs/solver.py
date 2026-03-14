"""
Set up the Kármán Vortex Street to benchmark numerical solvers with
"""

from ..core.config import KVSConfig
from ..core.result import KVSResult
from .methods import METHODS


def solve_kvs(
    config: KVSConfig,
    method: str = "fd",
    n_steps: int = 10_000,
    plot_every: int = 100,
    post_step=None,
    **kwargs,
) -> KVSResult:
    """
    Simulate the Kármán Vortex Street using the specified method.

    Args:
        config:      KVSConfig instance with physical and grid parameters.
        method:      Solver method, one of: 'fd', 'fe', 'lb'.
        n_steps:     Number of time steps to simulate.
        plot_every:  Visualisation interval (steps). 0 to disable.
        post_step:   Optional callback f(step, u) -> u applied after each step.
        **kwargs:    Additional arguments passed to the solver

    Returns:
        KVSResult with velocity fields and metadata.
    """
    if method not in METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(METHODS.keys())}"
        )

    solver = METHODS[method]
    return solver(
        config, n_steps=n_steps, plot_every=plot_every, post_step=post_step, **kwargs
    )
