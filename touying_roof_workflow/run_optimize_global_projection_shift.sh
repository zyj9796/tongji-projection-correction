#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

/usr/bin/python3 code/optimize_global_projection_shift.py "$@"
