#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

/usr/bin/python3 touying/code/optimize_local_projection_shifts.py "$@"
