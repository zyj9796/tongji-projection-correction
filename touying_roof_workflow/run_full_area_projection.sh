#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

/usr/bin/python3 code/generate_full_area_projection.py "$@"
