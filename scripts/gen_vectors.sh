#!/usr/bin/env bash
# Generate every golden vector, including the ones that are not kept in the repo.
#
# Usage: scripts/gen_vectors.sh
#
# 1. scripts/fetch_images.sh           calibration and held-out images
# 2. golden/gen_vectors.py             synthetic vectors, golden/vectors/*.hex (in the repo)
# 3. golden/gen_deit_forward.py        golden/vectors/deit/ and golden/vectors/deit_forward/
#                                      (not in the repo), and the accuracy report
# 4. golden/gen_mnist_forward.py       golden/vectors/mnist112/ (not in the repo) and
#                                      rtl/rom/mnist112/; downloads MNIST to golden/mnist/
#
# Step 3 needs torch and timm in .venv:
#   python3 -m venv --system-site-packages .venv && .venv/bin/pip install torch torchvision timm
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
"$root/scripts/fetch_images.sh"
python3 "$root/golden/gen_vectors.py"
"$root/.venv/bin/python" "$root/golden/gen_deit_forward.py"
# the 112 x 112 MNIST configuration; its probe, host/probe_mnist112.npz, is in the repo
# (host/train_probe.py trains it again)
"$root/.venv/bin/python" "$root/golden/gen_mnist_forward.py"
