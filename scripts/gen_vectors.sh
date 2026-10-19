#!/usr/bin/env bash
# Generate every golden vector, including the ones that are not kept in the repo.
#
# Usage: scripts/gen_vectors.sh
#
# 1. scripts/fetch_images.sh           calibration and held-out images
# 2. python/gen_vectors.py             synthetic vectors, python/vectors/*.hex
# 3. python/gen_deit_forward.py        python/vectors/deit/ and python/vectors/deit_forward/
#                                      (not in the repo), and the accuracy report
# 4. python/gen_mnist_forward.py       python/vectors/mnist112/ (not in the repo) and
#                                      rtl/rom/mnist112/; downloads MNIST to python/mnist/
#
# The tensor scales come from python/scales_deit224.json and
# python/scales_mnist112.json, which are in the repo, so the vectors are the same
# on every run. A generator run without --dump-only also prints the accuracy
# report; with --recalibrate it works the scales out again.
#
# Step 3 needs torch and timm in .venv:
#   python3 -m venv --system-site-packages .venv && .venv/bin/pip install torch torchvision timm
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
"$root/scripts/fetch_images.sh"
python3 "$root/python/gen_vectors.py"
"$root/.venv/bin/python" "$root/python/gen_deit_forward.py" --dump-only
# the 112 x 112 MNIST configuration; its probe, python/probe_mnist112.npz, is in the repo
# (python/train_probe.py trains it again)
"$root/.venv/bin/python" "$root/python/gen_mnist_forward.py" --dump-only
# the bytes python/run_image.py exchanges with the board for the test digit, for tb_axi_replay_mnist112
python3 "$root/python/run_image.py" --model --digit 0 --golden --clear --record
# the same with the weights in DDR, for tb_axi_replay_ddr_mnist112
python3 "$root/python/run_image.py" --model --digit 0 --golden --clear --ddr --record
# the same with one RUN of the run list, for tb_axi_replay_run_mnist112
python3 "$root/python/run_image.py" --model --digit 0 --golden --clear --run --record
# the external-memory bring-up check, for tb_nexys_top_ddr and tb_dma_ip
python3 "$root/python/ddr_check.py" --model --bytes 4096 --record
