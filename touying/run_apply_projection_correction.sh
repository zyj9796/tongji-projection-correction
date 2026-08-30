#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

/usr/bin/python3 touying/code/apply_projection_correction.py "$@"
