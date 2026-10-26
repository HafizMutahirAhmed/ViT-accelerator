#!/usr/bin/env bash
# Generate every golden vector the testbenches read; none of them is kept in the repo.
# The steps are in python/make_test_vectors.py, which needs torch, timm and torchvision in .venv.
#
# Usage: scripts/gen_vectors.sh
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
"$root/.venv/bin/python" "$root/python/make_test_vectors.py"
