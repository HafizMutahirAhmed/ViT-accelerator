"""Run one MNIST digit through the simulated design (no board) and print the digit it sees.

Usage:
  python3 python/sim_digit.py --digit N [--index K]     the K-th test image (default 0) labelled N
  python3 python/sim_digit.py --all-digits [--jobs J]   one image of each digit 0..9, J at a time

The image is loaded as python/run_image.py loads it (MNIST test set in
python/mnist), but chosen by label: --digit N --index K is the K-th test image
whose label is N, not test image N.

For each image:
  1. the software model of the board (python/engine_model.py) runs the whole
     112 x 112 MNIST program the way run_image.py does with --clear, reading
     back every stage that is still whole after its phase, and the bytes are
     recorded;
  2. scripts/sim.sh sim_digit_mnist112 plays those bytes to the RTL: the
     testbench is tb_axi_replay_mnist112 (axi_host_bridge, vit_axi_wrapper and
     the engine, i.e. nexys_top_axi without its UART modules), which checks
     every response byte, and tb_sim_digit_mnist112 around it writes down the
     bytes the design sent;
  3. those bytes are parsed by the same host code, which gives the simulated
     stages: the layer-2 output L1_add2 and the final LayerNorm (50 x 192 INT8
     each), the scores, and every other stage read back. Each is compared with
     the model's, entry by entry;
  4. the digit is the deployed classifier on the simulated output: the
     head_mean scores the simulated engine computed (the probe of
     probe_mnist112.npz on every row of the final LayerNorm), summed over the 49
     patch rows, as run_image.py does on a board.

The run directory is sim/sim_sim_digit_mnist112_SLOT<digit>/. One image takes
several minutes of simulation. Needs numpy and Vivado xsim on the PATH.
"""

import argparse
import concurrent.futures
import gzip
import os
import subprocess
import sys
import time

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HOST_DIR)
sys.path.insert(0, HOST_DIR)
import engine_model  # noqa: E402
import protocol  # noqa: E402
import run_image  # noqa: E402
import preprocess  # noqa: E402

TESTBENCH = "sim_digit_mnist112"
OUTPUT_STAGE = "L1_add2"            # the output of the two encoder layers (L1_x_out.hex), 50 x 192 INT8
PROBE_ACCURACY = "about 85.5%"
MINUTES_PER_IMAGE = 9                # one image: 6 to 9.6 min measured with Vivado 2022.2 xsim, depending on load


def test_labels():
    """Labels of the MNIST test set, from the same file run_image.mnist_test_digit reads."""
    path = os.path.join(HOST_DIR, "mnist", "MNIST", "raw", "t10k-labels-idx1-ubyte")
    opener, name = (open, path) if os.path.exists(path) else (gzip.open, path + ".gz")
    with opener(name, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)[8:]


def pick_image(digit, index):
    """(test set index, pixels, label) of the index-th test image labelled digit."""
    matches = np.flatnonzero(test_labels() == digit)
    if index >= len(matches):
        raise SystemExit(f"there are only {len(matches)} test images of digit {digit}")
    test_index = int(matches[index])
    pixels, label = run_image.mnist_test_digit(test_index)
    return test_index, pixels, label


class ReplayTransport:
    """Hands protocol.HostLink the bytes the simulated design sent, in order, and checks that the
    host sends the packets the simulation was given."""

    def __init__(self, packets, response):
        self.packets, self.response = packets, response
        self.packet_index = self.position = 0

    def write(self, data):
        if self.packet_index >= len(self.packets) or self.packets[self.packet_index] != data:
            raise protocol.ProtocolError(f"packet {self.packet_index} differs from the one the simulation got")
        self.packet_index += 1

    def read(self, count):
        data = self.response[self.position:self.position + count]
        self.position += len(data)
        return data


def stage_tensors(link, program, embed_in):
    """Runs the program over link; returns ({stage name: tensor}, error text or None)."""
    tensors = {}
    try:
        run_image.run(link, program, embed_in, check=lambda op, tensor: tensors.__setitem__(op["name"], tensor),
                      verbose=False, clear=True)
    except protocol.ProtocolError as error:
        return tensors, str(error)
    return tensors, None


def digit_of(tensors):
    """(digit, the 10 scores) of the deployed classifier: head_mean summed over the patch rows."""
    scores = tensors["head_mean"][1:].astype(np.int64).sum(0)
    return int(np.argmax(scores)), scores


def simulate(digit, index, program, quiet=False):
    """Runs one image; returns a dict with the results."""
    started = time.time()
    test_index, pixels, label = pick_image(digit, index)
    real = preprocess.im2col(preprocess.image_tensor(pixels, program.config["size"]))
    embed_in = np.clip(np.rint(real / program.config["input_scale"]), -128, 127).astype(np.int8)

    # 1. the software model, recorded
    model = protocol.HostLink(engine_model.ModelDevice(program.depths, os.path.join(
        REPO_DIR, program.config["rom_dir"])))
    model.record = []
    expected, error = stage_tensors(model, program, embed_in)
    if error:
        raise SystemExit(f"the software model failed: {error}")
    run_dir = os.path.join(REPO_DIR, "sim", f"sim_{TESTBENCH}_SLOT{digit}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "host_stream_tx.hex"), "w") as tx, \
            open(os.path.join(run_dir, "host_stream_rx.hex"), "w") as rx, \
            open(os.path.join(run_dir, "host_stream_packets.txt"), "w") as packets:
        for packet, response in model.record:
            tx.write("".join(f"{byte:02x}\n" for byte in packet))
            rx.write("".join(f"{byte:02x}\n" for byte in response))
            packets.write(f"{len(packet)} {len(response)} {int(packet[1] == protocol.START)}\n")
    dump = os.path.join(run_dir, "dut_response.hex")
    if os.path.exists(dump):
        os.remove(dump)

    # 2. the simulation
    if not quiet:
        print(f"digit {digit}, index {index}: MNIST test image {test_index}, label {label}; "
              f"{len(model.record)} packets, simulating (several minutes) ...", flush=True)
    sim = subprocess.run([os.path.join(REPO_DIR, "scripts", "sim.sh"), TESTBENCH, f"SLOT={digit}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = os.path.join(REPO_DIR, "sim", f"sim_{TESTBENCH}_SLOT{digit}.log")
    verdict = "no PASS/FAIL line"
    if os.path.exists(log):
        lines = [line.strip() for line in open(log) if line.startswith(("PASS", "FAIL"))]
        verdict = lines[-1] if lines else verdict

    # 3. the simulated stages
    response = b""
    if os.path.exists(dump):
        with open(dump) as f:
            # an unknown byte (xx) is counted as 0: it keeps its place and shows up as a mismatch
            response = bytes(0 if "x" in line.lower() else int(line, 16) for line in f if line.strip())
    transport = ReplayTransport([packet for packet, _ in model.record], response)
    simulated, parse_error = stage_tensors(protocol.HostLink(transport, retries=1), program, embed_in)

    stages, mismatches, first = [], 0, None
    for name, want in expected.items():
        got = simulated.get(name)
        if got is None:
            stages.append((name, None))
            continue
        wrong = got != want
        stages.append((name, int(wrong.sum())))
        mismatches += int(wrong.sum())
        if wrong.any() and first is None:
            row, col = (int(i) for i in np.argwhere(wrong)[0])
            first = (name, row, col, int(got[row, col]), int(want[row, col]))
    missing = [name for name, count in stages if count is None]
    exact = (sim.returncode == 0 and verdict.startswith("PASS") and parse_error is None and not missing
             and mismatches == 0)
    result = dict(digit=digit, index=index, test_index=test_index, label=label, exact=exact,
                  mismatches=mismatches, first=first, missing=missing, parse_error=parse_error,
                  verdict=verdict, returncode=sim.returncode, log=log, stages=stages,
                  output_entries=expected[OUTPUT_STAGE].size, seconds=time.time() - started)
    result["model_digit"], result["model_scores"] = digit_of(expected)
    if "head_mean" in simulated:
        result["sim_digit"], result["sim_scores"] = digit_of(simulated)
    else:
        result["sim_digit"], result["sim_scores"] = None, None
    return result


def report(r):
    print(f"digit {r['digit']}, index {r['index']}: MNIST test image {r['test_index']}, true label {r['label']}")
    print(f"  testbench: {r['verdict']}   ({r['log']})")
    if r["parse_error"]:
        print(f"  reading the simulated responses stopped: {r['parse_error']}")
    for name, count in r["stages"]:
        if name in (OUTPUT_STAGE, "final_ln", "head_mean", "head_class") or count:
            print(f"    stage {name:13s} " + ("not read back" if count is None else
                                              "equal" if count == 0 else f"{count} entries differ"))
    print(f"  {len(r['stages'])} stages compared; {OUTPUT_STAGE} is the 50 x 192 INT8 output of the two layers")
    if r["exact"]:
        print(f"  BIT-EXACT PASS: 0 mismatching values (output {r['output_entries']} entries and every other stage)")
    else:
        print(f"  BIT-EXACT FAIL: {r['mismatches']} mismatching values"
              + (f", {len(r['missing'])} stages missing" if r["missing"] else ""))
        if r["first"]:
            name, row, col, got, want = r["first"]
            print(f"  first mismatch: stage {name}, row {row}, column {col}: simulated {got}, model {want}")
    if r["sim_scores"] is not None:
        print(f"  simulated scores (head_mean summed over the patch rows): {r['sim_scores'].tolist()}")
        print(f"  simulated prediction: {r['sim_digit']}   true label: {r['label']}")
    else:
        print("  simulated prediction: none, the scores were not read back")
    print(f"  Python model prediction: {r['model_digit']}   ("
          + ("agrees with the simulation" if r["sim_digit"] == r["model_digit"] else "DIFFERS from the simulation")
          + ")")
    if r["exact"] and r["sim_digit"] != r["label"]:
        print(f"  note: the prediction is wrong but the simulation is bit-exact with the model; that is the "
              f"probe's error ({PROBE_ACCURACY} accurate), not the hardware's")
    print(f"  wall time {r['seconds'] / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--digit", type=int, choices=range(10))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--all-digits", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="with --all-digits: simulations at once")
    parser.add_argument("--program", default=os.path.join(HOST_DIR, "vectors", "mnist112"))
    args = parser.parse_args()
    if (args.digit is None) == (not args.all_digits):
        parser.error("give --digit N or --all-digits")
    program = run_image.Program(args.program)

    if not args.all_digits:
        r = simulate(args.digit, args.index, program)
        report(r)
        return 0 if r["exact"] else 1

    estimate = MINUTES_PER_IMAGE * -(-10 // args.jobs)
    print(f"10 images, {args.jobs} at a time; about {MINUTES_PER_IMAGE} min each (xsim), "
          f"so roughly {estimate} min in all", flush=True)
    if estimate > 30:
        print("WARNING: the table will take more than 30 minutes", flush=True)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda d: simulate(d, 0, program, quiet=True), range(10)))
    print()
    print("digit  test image  simulated  Python  bit-exact  wall time")
    for r in results:
        print(f"{r['digit']:5d}  {r['test_index']:10d}  {str(r['sim_digit']):>9s}  {r['model_digit']:6d}  "
              f"{'PASS' if r['exact'] else 'FAIL':>9s}  {r['seconds'] / 60:6.1f} min")
    for r in results:
        if not r["exact"]:
            print(f"digit {r['digit']}: {r['verdict']}; {r['mismatches']} mismatching values"
                  + (f", first: stage {r['first'][0]} row {r['first'][1]} column {r['first'][2]}" if r["first"]
                     else "") + (f"; {r['parse_error']}" if r["parse_error"] else "") + f"; see {r['log']}")
        if r["sim_digit"] != r["model_digit"]:
            print(f"digit {r['digit']}: the simulated and the Python predictions differ")
        elif r["exact"] and r["sim_digit"] != r["label"]:
            print(f"digit {r['digit']}: predicted {r['sim_digit']}, wrong, but bit-exact: the probe's error "
                  f"({PROBE_ACCURACY} accurate), not the hardware's")
    exact = sum(r["exact"] for r in results)
    right = sum(r["sim_digit"] == r["label"] for r in results)
    print(f"bit-exact {exact} of 10; simulated predictions right {right} of 10; "
          f"total wall time {(time.time() - started) / 60:.1f} min")
    return 0 if exact == 10 else 1


if __name__ == "__main__":
    sys.exit(main())
