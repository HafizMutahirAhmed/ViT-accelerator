"""Make every golden vector the testbenches read (the simulation sets of vivado/vit.xpr).

Usage: .venv/bin/python python/make_test_vectors.py     (needs torch, timm and torchvision)

1. gen_vectors.py                    synthetic vectors of the core blocks, python/vectors/*.hex
2. make_model.py                     the model configurations the testbenches use, in
                                     python/models/<board>_L<layers>/ (MNIST is downloaded to
                                     python/mnist/ on the first run):
     --board opal_kelly --layers 2   224 x 224: tb_encoder_program, tb_matmul_engine_opal_kelly
                                     (matmuls/) and the non-linear unit testbenches
     --board nexys --layers 2        112 x 112: tb_encoder_program_nexys and the host streams
     --board nexys --layers 6        a deeper model, for tb_axi_replay_run_nexys_L6
     --board nexys --layers 12       all 12 layers, for tb_axi_replay_run_nexys_L12
3. the bytes the host software exchanges with the board, against the software model of the
   board, for the replay testbenches: run_image.py (tb_axi_replay_run, --step for
   tb_axi_replay_step) and ddr_check.py (tb_nexys_top_ddr, tb_dma_ip)

The tensor scales (python/calibration/input_<size>_scales.json) and the digit classifiers
(python/calibration/input_<size>_L<layers>_probe.npz) are in the repo, so the vectors are the same
on every run. The torch environment, once:
  python3 -m venv --system-site-packages .venv && .venv/bin/pip install torch torchvision timm
"""
import os
import subprocess
import sys

HOST_DIR = os.path.dirname(os.path.abspath(__file__))


def run(script, *arguments):
    subprocess.run([sys.executable, os.path.join(HOST_DIR, script), *arguments], check=True)


run("gen_vectors.py")
for board, layers in (("opal_kelly", 2), ("nexys", 2), ("nexys", 6), ("nexys", 12)):
    run("make_model.py", "--board", board, "--layers", str(layers))
for layers in (2, 6, 12):
    run("run_image.py", "--model", "--digit", "0", "--golden", "--clear", "--record",
        "--model-dir", os.path.join(HOST_DIR, "models", f"nexys_L{layers}"))
run("run_image.py", "--model", "--digit", "0", "--golden", "--clear", "--step", "--record")
run("ddr_check.py", "--model", "--bytes", "4096", "--record")
print("make_test_vectors: every vector is made")
