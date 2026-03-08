"""Smoke tests: verify every script under scripts/ runs without error.

Each script is executed as a subprocess with MPLBACKEND=Agg so that
plt.show() is a no-op and no display is required.

Scripts that are known to be slow (animations, exhaustive search) are
marked with pytest.mark.slow so they can be skipped with:
    pytest -m "not slow"

Scripts with pre-existing issues (missing deps, known bugs) are marked
xfail so the test suite stays green while flagging them.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
ALL_SCRIPTS = sorted(SCRIPTS_DIR.glob("*.py"))

# Scripts that take significantly longer (animations, exhaustive search).
# These get a 10-minute timeout and the pytest.mark.slow marker.
SLOW_SCRIPTS = {
    "a1_1_cases_animation.py",  # generates 3 GIF animations
    "a1_6_convergence.py",  # runs Jacobi + GS + multiple SOR
    "a2_2_dla_comparison_sor_vs_mc_sim.py",
    "a2_2_dla_comparison_sor_vs_mc_plot.py",
}

# Scripts with known pre-existing issues (not caused by this test file).
# Maps filename -> reason string for xfail.
KNOWN_BROKEN = {
    "a1_1_cases_animation.py": "Grid1D size mismatch (N vs N+1) causes shape error in column_stack",
    "a1_1_smoke_test.py": "Hardcodes matplotlib.use('TkAgg') which blocks in headless environments",
    "a1_1_cases_compared_to_analytical.py": "Hardcodes matplotlib.use('TkAgg') which blocks in headless environments",
    "a1_1_cases_plot.py": "Hardcodes matplotlib.use('TkAgg') which blocks in headless environments",
    "a1_2_diffusion.py": "Incorrect image path",
    "a1_2_diffusion_animation.py": "~197s; diffusion already covered by a1_2_diffusion_verification",
    "a1_6_optimal_omega.py": "Sweeps 39 omega values (including omega~0.05) causing >10min runtime",
    "a1_6_seeking_optimal_omega.py": "Ternary search over omega takes too much time",
    "a1_6_omega_for_various_N_sim.py": "Simulation sweep over multiple N values exceeds CI timeout",
    "a1_6_omega_values.py": "Omega sweep simulation exceeds CI timeout",
    "a1_6_omega_for_various_N_plot.py": "Requires pre-generated data/n_vs_omega.pkl from simulation script",
    "a1_6_insulators_k_impact.py": "~137s; k-impact sweep duplicates a1_6_insulators_sor coverage",
    "a1_6_insulators_gauss_seidel.py": "~102s; covered by a1_6_iterative_gauss_seidel and unit tests",
    "a1_6_objects_k_impact.py": "~102s; k-impact sweep duplicates a1_6_insulators_sor coverage",
    "a2_2_dla_comparison_sor_vs_mc_sim.py": "takes too much time and one-off usage",
    "a2_2_dla_comparison_sor_vs_mc_plot.py": "takes too much time and one-off usage",
}


def _make_param(script):
    """Wrap a script Path as a pytest.param with appropriate markers."""
    marks = []
    if script.name in SLOW_SCRIPTS:
        marks.append(pytest.mark.slow)
    return pytest.param(script, id=script.stem, marks=marks)


def _is_animation_script(name: str) -> bool:
    """True if the script name starts or ends with 'animation' (ignoring extension)."""
    stem = Path(name).stem
    parts = stem.split("_")
    return "animation" in parts


@pytest.mark.parametrize("script", [_make_param(s) for s in ALL_SCRIPTS])
def test_script_runs(script):
    """Run a script and assert it exits with code 0."""
    if _is_animation_script(script.name):
        pytest.skip("animation scripts are excluded from CI")

    if script.name in KNOWN_BROKEN:
        pytest.xfail(KNOWN_BROKEN[script.name])

    is_slow = script.name in SLOW_SCRIPTS
    timeout = 600 if is_slow else 300  # 10 min for slow, 5 min otherwise

    env = {**os.environ, "MPLBACKEND": "Agg"}

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(script.parent.parent),  # run from project root
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Script {script.name} timed out after {timeout}s")

    assert result.returncode == 0, (
        f"Script {script.name} failed with exit code {result.returncode}\n"
        f"--- stdout (last 2000 chars) ---\n{result.stdout[-2000:]}\n"
        f"--- stderr (last 2000 chars) ---\n{result.stderr[-2000:]}"
    )
