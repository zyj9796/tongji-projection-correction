#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

/usr/bin/python3 touying/code/generate_full_area_projection.py "$@"
