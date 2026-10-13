"""Run one image through the accelerator and print the digit it sees.

Usage:
  python3 python/run_image.py --port /dev/ttyUSB1 --ping             is the board there?
  python3 python/run_image.py --port /dev/ttyUSB1 --digit 0          a board on a serial port
  python3 python/run_image.py --model --digit 0                      the software model of the board
  python3 python/run_image.py --port /dev/ttyUSB1 --png digit.png --check

  --ping         only ask the board for its status and print it: a link check that takes
                 a moment. Exit code 0 if the board answered and its error flag is clear.
  --digit N      MNIST test digit N (from python/mnist, as torchvision downloads it)
  --png FILE     any image file; it is made greyscale and resized to 112 x 112
  --baud N       serial speed, default 921600
  --check        after every phase read back every op output that is still whole
                 and compare it with the software model fed the same bytes
  --golden       with --model: compare the stages with the golden files of
                 python/vectors/mnist112 instead (only right for the test digit)
  --program DIR  the program directory, default python/vectors/mnist112
  --clear        write zeros to every pool buffer first. The bank words past a
                 tensor's last row are never written by an op; on a board they hold
                 what the block RAM was configured with (zeros), in a simulation
                 they are unknown until written
  --record       with --model: write every byte sent and received to the program
                 directory (host_stream_tx.hex, host_stream_rx.hex, one byte per
                 line, and host_stream_packets.txt: bytes sent, bytes received and
                 whether it is a START, per packet) for tb_host_replay_mnist112, which plays
                 them to the RTL

The program directory comes from python/gen_mnist_forward.py: the descriptor
table, the phases, what to load before each phase and where every stage is.
This script quantizes the image, builds the im2col matrix, and follows
docs/HOST_PROTOCOL.md: descriptors, then per phase the loads, START and
STATUS until idle. The scores are read from the last phase's outputs: the
host sums the 49 patch rows of head_mean for the answer and also reports the
class row's own classifier, head_class.

Needs numpy and Pillow; pyserial for a board. No torch.
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HOST_DIR)
sys.path.insert(0, HOST_DIR)
import preprocess  # noqa: E402
import protocol  # noqa: E402

LANES = 8


def mnist_test_digit(index):
    """(pixels 28 x 28 uint8, label) of MNIST test digit `index`."""
    raw = os.path.join(HOST_DIR, "mnist", "MNIST", "raw")

    def load(name, header):
        path = os.path.join(raw, name)
        opener = open if os.path.exists(path) else gzip.open
        with opener(path if os.path.exists(path) else path + ".gz", "rb") as f:
            return np.frombuffer(f.read(), dtype=np.uint8)[header:]

    images = load("t10k-images-idx3-ubyte", 16).reshape(-1, 28, 28)
    labels = load("t10k-labels-idx1-ubyte", 8)
    return images[index], int(labels[index])


def row_bank_bytes(tensor):
    """rows x cols INT8 tensor as WRITE_BUFFER items: one address per (row tile, column), 8 banks."""
    rows, cols = tensor.shape
    tiles = -(-rows // LANES)
    padded = np.zeros((tiles * LANES, cols), dtype=np.int8)
    padded[:rows] = tensor
    return padded.reshape(tiles, LANES, cols).transpose(0, 2, 1).tobytes()


def read_hex_lines(path):
    with open(path) as f:
        return [int(line, 16) for line in f if line.strip()]


class Program:
    """The files python/layer_program.py wrote."""

    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, "host_config.json")) as f:
            self.config = json.load(f)
        with open(os.path.join(directory, "program_plan.txt")) as f:
            self.depths = [int(word) for word in f.read().split()][1:]
        self.phases = [(name, int(first), int(count)) for name, first, count in
                       (line.split() for line in open(os.path.join(directory, "program_phases.txt")))]
        self.ops = []
        for line in open(os.path.join(directory, "program_ops.txt")):
            entry, phase, name, kind, buffer_id, layout, base, stride, rows, cols, file_name = line.split()
            self.ops.append(dict(entry=int(entry), phase=phase, name=name, kind=kind, buffer=int(buffer_id),
                                 layout=layout, base=int(base), stride=int(stride), rows=int(rows),
                                 cols=int(cols), file=file_name))
        self.survivors = {int(line) for line in open(os.path.join(directory, "program_survivors.txt"))}
        self.loads = [(phase, kind, int(target), int(base), int(words), file_name) for
                      phase, kind, target, base, words, file_name in
                      (line.split() for line in open(os.path.join(directory, "program_loads.txt")))]
        words = read_hex_lines(os.path.join(directory, "program_desc.hex"))
        self.descriptor_bytes = b"".join(word.to_bytes(4, "little") for word in words)

    def load_bytes(self, kind, file_name):
        size = {"buffer": 8, "bias": 32, "layernorm": 6}[kind]
        return b"".join(value.to_bytes(size, "little") for value in read_hex_lines(
            os.path.join(self.directory, file_name)))


def region(op):
    """(first address, number of addresses) of an op's output in its buffer."""
    outer, inner = (op["rows"], op["cols"]) if op["layout"] == "row" else (op["cols"], op["rows"])
    return op["base"], (-(-outer // LANES) - 1) * op["stride"] + inner


def tensor_of(op, data):
    """The rows x cols tensor of an op from the bytes READ_BUFFER returned for its region."""
    banks = np.frombuffer(data, dtype=np.int8).reshape(-1, LANES)
    row, col = np.arange(op["rows"])[:, None], np.arange(op["cols"])[None, :]
    if op["layout"] == "row":
        return banks[(row // LANES) * op["stride"] + col, row % LANES]
    return banks[(col // LANES) * op["stride"] + row, col % LANES]


def run(link, program, embed_in, check=None, verbose=True, clear=False):
    """Loads and runs every phase. check(op, tensor) is called for every stage that can be read back.
    Returns (scores of head_mean 50 x 10, scores of head_class 1 x 10, engine cycles)."""
    link.clear()
    if clear:
        for buffer_id, depth in enumerate(program.depths):
            link.write(protocol.WRITE_BUFFER, buffer_id, 0, bytes(8 * depth))
    link.write(protocol.WRITE_DESCRIPTOR, 0, 0, program.descriptor_bytes)
    cycles = 0
    outputs = {}
    for phase, first, count in program.phases:
        for load_phase, kind, target, base, _words, file_name in program.loads:
            if load_phase != phase:
                continue
            data = row_bank_bytes(embed_in) if file_name == "embed_in_image.hex" else program.load_bytes(
                kind, file_name)
            code = {"buffer": protocol.WRITE_BUFFER, "bias": protocol.WRITE_BIAS,
                    "layernorm": protocol.WRITE_LAYERNORM}[kind]
            link.write(code, target, base, data)
        link.start(first)
        cycles += link.wait_idle()
        for op in program.ops[first:first + count]:
            wanted = op["name"] in ("head_mean", "head_class")
            if (check and op["entry"] in program.survivors) or wanted:
                tensor = tensor_of(op, link.read_buffer(op["buffer"], *region(op)))
                outputs[op["name"]] = tensor
                if check:
                    check(op, tensor)
        if verbose:
            print(f"  phase {phase:9s} {count:2d} ops done", flush=True)
    return outputs["head_mean"], outputs["head_class"], cycles


def ping(args):
    """STATUS, once: is there a board that speaks the protocol, and is it happy?"""
    if args.model:
        import engine_model
        program = Program(args.program)
        link = protocol.HostLink(engine_model.ModelDevice(program.depths, os.path.join(
            REPO_DIR, program.config["rom_dir"])))
        where = "the software model"
    else:
        import transports
        where = f"{args.port} at {args.baud} baud"
        try:
            link = protocol.HostLink(transports.SerialTransport(args.port, args.baud, timeout=2.0))
        except Exception as error:   # the port does not exist, is busy, or needs the dialout group
            print(f"PING FAIL: cannot open {args.port}: {error}")
            return 1
    try:
        flags, cycles = link.status()
    except protocol.ProtocolError as error:
        print(f"PING FAIL: no valid answer from {where}: {error}")
        return 1
    busy, done, error_flag = (bool(flags & bit) for bit in (protocol.FLAG_BUSY, protocol.FLAG_DONE,
                                                             protocol.FLAG_ERROR))
    print(f"PING OK: {where} answered")
    print(f"  busy  {int(busy)}   the engine is {'running' if busy else 'idle'}")
    print(f"  done  {int(done)}   {'a run has finished since the last START or CLEAR' if done else 'no finished run'}")
    print(f"  error {int(error_flag)}   {'a command was refused or had a wrong checksum; CLEAR resets this' if error_flag else 'no protocol error'}")
    print(f"  busy cycles of the last run: {cycles}")
    return 1 if error_flag else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--digit", type=int)
    parser.add_argument("--png")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--ping", action="store_true")
    parser.add_argument("--program", default=os.path.join(HOST_DIR, "vectors", "mnist112"))
    args = parser.parse_args()
    if (args.port is None) == (not args.model):
        parser.error("give --port or --model")
    if args.ping:
        return ping(args)
    if (args.digit is None) == (args.png is None):
        parser.error("give --digit or --png (or --ping)")

    program = Program(args.program)
    label = None
    if args.png:
        pixels = preprocess.load_png(args.png)
    else:
        pixels, label = mnist_test_digit(args.digit)
    real = preprocess.im2col(preprocess.image_tensor(pixels, program.config["size"]))
    embed_in = np.clip(np.rint(real / program.config["input_scale"]), -128, 127).astype(np.int8)

    def model_link():
        import engine_model
        return protocol.HostLink(engine_model.ModelDevice(program.depths, os.path.join(
            REPO_DIR, program.config["rom_dir"])))

    if args.model:
        link = model_link()
    else:
        import transports
        link = protocol.HostLink(transports.SerialTransport(args.port, args.baud))

    failures = []
    expected = {}
    if args.check and not args.model:
        # the software model, fed the same image, says what every stage must be
        run(model_link(), program, embed_in, check=lambda op, tensor: expected.__setitem__(op["name"], tensor),
            verbose=False)

    def check(op, tensor):
        if args.golden:
            want = np.array(read_hex_lines(os.path.join(program.directory, op["file"])), dtype=np.uint8) \
                .astype(np.int8).reshape(op["rows"], op["cols"])
        else:
            want = expected[op["name"]]
        wrong = int((tensor != want).sum())
        if wrong:
            first = int(np.argmax((tensor != want).reshape(-1)))
            failures.append(op["name"])
            print(f"  stage {op['name']:13s} FAIL  {wrong} of {tensor.size} entries wrong, first mismatch index {first}")
        else:
            print(f"  stage {op['name']:13s} PASS  {tensor.size} entries")

    if args.record:
        link.record = []
    started = time.time()
    checking = (args.check and not args.model) or (args.model and args.golden)
    mean_scores, class_scores, cycles = run(link, program, embed_in, check if checking else None,
                                            clear=args.clear)
    elapsed = time.time() - started

    patch_sum = mean_scores[1:].astype(np.int64).sum(0)
    answer, class_answer = int(np.argmax(patch_sum)), int(np.argmax(class_scores[0]))
    print(f"summed scores of the patch rows: {patch_sum.tolist()}")
    print(f"class-row scores:                {class_scores[0].tolist()}")
    print(f"digit: {answer}   (class-row classifier: {class_answer})"
          + ("" if label is None else f"   label: {label}"))
    print(f"engine cycles {cycles}; {link.bytes_sent} bytes sent, {link.bytes_received} received, "
          f"{elapsed:.1f} s")
    if args.record:
        with open(os.path.join(program.directory, "host_stream_tx.hex"), "w") as tx, \
                open(os.path.join(program.directory, "host_stream_rx.hex"), "w") as rx, \
                open(os.path.join(program.directory, "host_stream_packets.txt"), "w") as packets:
            for packet, response in link.record:
                tx.write("".join(f"{byte:02x}\n" for byte in packet))
                rx.write("".join(f"{byte:02x}\n" for byte in response))
                packets.write(f"{len(packet)} {len(response)} {int(packet[1] == protocol.START)}\n")
        print(f"recorded {len(link.record)} packets to {program.directory}/host_stream_*")
    if checking:
        print("CHECK FAIL: " + ", ".join(failures) if failures else "CHECK PASS: every stage read back matches")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
