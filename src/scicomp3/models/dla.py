"""PDE-based Diffusion Limited Aggregation (DLA) model.

Uses the steady-state diffusion equation (Laplace equation) to compute the
concentration field, which drives probabilistic cluster growth.

Grid convention (same as rest of scicomp3):
    - Array shape: (N+1, N+1)
    - axis 0 (rows)   → x-direction, periodic
    - axis 1 (cols)   → y-direction, Dirichlet (bottom c=0, top c=1)
    - (i, j) corresponds to x=i/N, y=j/N

Growth algorithm (Assignment 2.1):
    1. Start with a seed cluster at the bottom of the domain.
    2. Solve the Laplace equation: cluster = sink (c=0), top boundary = source (c=1).
    3. Find growth candidates: interior non-cluster points adjacent to the cluster.
    4. Compute growth probabilities: pg(i,j) = c^η / sum(c^η over candidates).
    5. Select one candidate at random (weighted by pg) and add it to the cluster.
    6. Repeat, warm-starting SOR from the previous concentration field.

References:
    Assignment Set 2, section 2.1 — Diffusion Limited Aggregation
"""

import numpy as np

from ..bvp.solver import solve_bvp
from ..bvp.omega import get_optimal_omega


# ---------------------------------------------------------------------------
# Helper functions (also usable standalone / in tests)
# ---------------------------------------------------------------------------


def find_growth_candidates(cluster_mask):
    """Find all growth-candidate sites adjacent to the cluster.

    A growth candidate is an interior point that
      * is not in the cluster, and
      * has at least one N/E/S/W neighbor that *is* in the cluster.

    The top and bottom boundary rows (j=0, j=N) are always excluded, since
    they are fixed by Dirichlet boundary conditions.

    Periodicity in x is handled implicitly via np.roll (same convention as
    the BVP solver).

    Args:
        cluster_mask: Boolean array of shape (N+1, N+1).  True = cluster site.

    Returns:
        candidates: Boolean array, same shape as cluster_mask.
    """
    has_cluster_neighbor = (
        np.roll(cluster_mask, -1, axis=0)  # right in x (periodic)
        | np.roll(cluster_mask, 1, axis=0)  # left  in x (periodic)
        | np.roll(cluster_mask, -1, axis=1)  # up    in y
        | np.roll(cluster_mask, 1, axis=1)  # down  in y
    )
    candidates = (~cluster_mask) & has_cluster_neighbor
    candidates[:, 0] = False  # exclude bottom boundary (j=0)
    candidates[:, -1] = False  # exclude top boundary    (j=N)
    return candidates


def compute_growth_probabilities(concentration, candidates, eta=1.0):
    """Compute normalised growth probabilities for the candidate sites.

    pg(i, j) = c(i, j)^η  /  Σ  c(i', j')^η
                               candidates

    The values are in the same order as ``np.argwhere(candidates)``.

    Args:
        concentration: Float array (N+1, N+1).
        candidates:    Boolean array (N+1, N+1).
        eta:           Growth exponent (default 1.0).
                       η=0 → Eden model (uniform), η=1 → standard DLA,
                       η>1 → more open/lightning-like cluster.

    Returns:
        pg: 1-D float array of normalised probabilities (sum = 1).
    """
    c_vals = np.clip(concentration[candidates], 0.0, None)
    pg = c_vals**eta
    total = pg.sum()
    if total == 0.0:
        # Fallback: uniform probability (shouldn't normally happen)
        pg = np.ones(len(pg)) / len(pg)
    else:
        pg /= total
    return pg


def _bvp_bc(k, c):
    """Enforce Dirichlet BCs for the DLA diffusion problem."""
    c[:, 0] = 0.0  # bottom boundary  y=0  →  c = 0
    c[:, -1] = 1.0  # top boundary     y=1  →  c = 1
    return c


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def run_dla(
    N=100,
    n_steps=200,
    eta=1.0,
    omega=None,
    tol=1e-4,
    max_iter=2_000,
    rng=None,
    method="sor",
):
    """Run a PDE-based DLA simulation on an (N+1)×(N+1) grid.

    The Laplace equation is solved with SOR after each growth step.  The
    previous concentration field is used as a warm start so that SOR
    converges in very few iterations (the field changes by only one new sink
    per step).

    Args:
        N:        Grid size; the domain has (N+1)×(N+1) points.
        n_steps:  Number of growth steps (particles added to cluster).
        eta:      Growth exponent (see ``compute_growth_probabilities``).
        omega:    SOR relaxation parameter.  Defaults to the theoretical
                  optimal for an N×N Laplace problem.
        tol:      BVP convergence tolerance (default 1e-4 — slightly relaxed
                  vs. steady-state solve, since only relative field shape
                  matters for growth probabilities).
        max_iter: Maximum SOR iterations per growth step (caps cost when
                  warm-started; convergence is usually fast).
        rng:      NumPy ``Generator`` for reproducibility.  If None, a new
                  ``default_rng()`` is used.
        method:   BVP solver method name (default "sor").  Also accepts
                  "sor_redblack" (vectorised) or "sor_numba" (JIT-compiled).

    Returns:
        cluster_mask:  Boolean array (N+1, N+1).  True at every site that
                       belongs to the grown cluster.
        concentration: Float array (N+1, N+1).  Concentration field from the
                       last BVP solve.
        n_iters:       List of BVP iteration counts, one per growth step.
    """
    if rng is None:
        rng = np.random.default_rng()
    if omega is None:
        omega = get_optimal_omega(N)

    # --- Initial cluster: single seed at the bottom-centre ---
    cluster_mask = np.zeros((N + 1, N + 1), dtype=bool)
    cluster_mask[N // 2, 1] = True  # j=1: first interior row above bottom BC

    # --- Initial concentration: linear gradient c(y) = y ---
    # (analytical solution for the empty domain, eq. 5 in the assignment)
    y_coords = np.linspace(0.0, 1.0, N + 1)
    concentration = np.broadcast_to(y_coords, (N + 1, N + 1)).copy()

    n_iters = []

    for _ in range(n_steps):
        sink_coords = np.argwhere(cluster_mask)

        # Solve Laplace equation — warm-started from previous concentration
        result = solve_bvp(
            concentration,
            method=method,
            tol=tol,
            max_iter=max_iter,
            post_step=_bvp_bc,
            sink_coordinates=sink_coords,
            omega=omega,
        )
        concentration = result.y
        n_iters.append(result.n_iter)

        # Find growth candidates
        candidates = find_growth_candidates(cluster_mask)
        if not np.any(candidates):
            break  # cluster reached top boundary — stop early

        # Select one candidate weighted by growth probability
        candidate_indices = np.argwhere(candidates)
        pg = compute_growth_probabilities(concentration, candidates, eta)
        choice_idx = rng.choice(len(candidate_indices), p=pg)
        new_point = tuple(candidate_indices[choice_idx])
        cluster_mask[new_point] = True

    return cluster_mask, concentration, n_iters
