"""PDE right-hand-side implementations."""

from .wave import wave1d_rhs
from .diffusion import (
    diffusion2d_rhs,
    apply_diffusion_bc,
    diffusion_stable_dt,
    analytical_solution,
)
from .gray_scott import (
    gray_scott_rhs,
    gray_scott_initial_conditions,
    gray_scott_stable_dt,
)

__all__ = [
    "wave1d_rhs",
    "diffusion2d_rhs",
    "apply_diffusion_bc",
    "diffusion_stable_dt",
    "analytical_solution",
    "gray_scott_rhs",
    "gray_scott_initial_conditions",
    "gray_scott_stable_dt",
]
