#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

/usr/bin/python3 touying/code/plot_corrected_projection.py "$@"
