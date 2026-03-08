#!/usr/bin/env bash
# Generate a flamegraph SVG for pure-Python SOR DLA using py-spy.
#
# Output: images/profiling/a2_1_dla_flamegraph.svg
#
# Usage:
#   bash scripts/a2_1_dla_by_sor_flamegraph.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
PY_SPY="${PROJECT_DIR}/.venv/bin/py-spy"

OUT_DIR="${PROJECT_DIR}/images/profiling"
mkdir -p "$OUT_DIR"
SVG="${OUT_DIR}/a2_1_dla_flamegraph.svg"

export MPLBACKEND=Agg

echo "Recording flamegraph with py-spy..."
"$PY_SPY" record -o "$SVG" --rate 100 -- \
    "$PYTHON" "$SCRIPT_DIR/a2_1_dla_by_sor.py"

echo "Saved → ${SVG}"
