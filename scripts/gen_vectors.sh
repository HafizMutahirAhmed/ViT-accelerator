#!/usr/bin/env bash
# Generate every golden vector the testbenches read; none of them is kept in the repo.
#
# Usage: scripts/gen_vectors.sh
#
# 1. python/gen_vectors.py             synthetic vectors of the core blocks, python/vectors/*.hex
# 2. python/make_model.py              the model configurations the testbenches use, in
#                                      python/models/<board>_L<layers>/ (MNIST is downloaded
#                                      to python/mnist/ on the first run):
#      --board opal_kelly --layers 2   224 x 224: tb_encoder_program, tb_matmul_engine_opal_kelly
#                                      (matmuls/) and the non-linear unit testbenches
#      --board nexys --layers 2        112 x 112: tb_encoder_program_nexys and the host streams
#      --board nexys --layers 6        a deeper model, for tb_axi_replay_run_nexys_L6
# 3. the bytes the host software exchanges with the board, against the software model of the
#    board, for the replay testbenches: run_image.py (tb_axi_replay_run, --step for
#    tb_axi_replay_step) and ddr_check.py (tb_nexys_top_ddr, tb_dma_ip)
#
# The tensor scales (python/calibration/<board>_scales.json) and the digit classifiers
# (python/calibration/<board>_L<layers>_probe.npz) are in the repo, so the vectors are the same on
# every run. Step 2 needs torch, timm and torchvision in .venv:
#   python3 -m venv --system-site-packages .venv && .venv/bin/pip install torch torchvision timm
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
python3 "$root/python/gen_vectors.py"
for model in "opal_kelly 2" "nexys 2" "nexys 6"; do
    set -- $model
    "$root/.venv/bin/python" "$root/python/make_model.py" --board "$1" --layers "$2"
done
for layers in 2 6; do
    python3 "$root/python/run_image.py" --model --digit 0 --golden --clear --record \
        --model-dir "$root/python/models/nexys_L$layers"
done
python3 "$root/python/run_image.py" --model --digit 0 --golden --clear --step --record
python3 "$root/python/ddr_check.py" --model --bytes 4096 --record
