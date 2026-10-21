#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${SPARK_PYTHON:-python}" "$ROOT/scrpits/run_unitree_g1_reconstruction_teleop.py" "$@"
