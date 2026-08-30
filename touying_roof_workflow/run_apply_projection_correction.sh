#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

/usr/bin/python3 code/apply_projection_correction.py "$@"
