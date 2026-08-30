#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

/usr/bin/python3 touying/code/simulate_dsm_sar_correction.py "$@"
