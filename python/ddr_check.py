"""Bring-up check of the external memory: the DDR alone, then the DMA into the engine, then RUN.

Usage:
  python3 python/ddr_check.py --port /dev/ttyUSB1               a board running nexys_top_ddr
  python3 python/ddr_check.py --port /dev/ttyUSB1 --bytes 1048576
  python3 python/ddr_check.py --model                           the software model of the board
  python3 python/ddr_check.py --model --bytes 4096 --record     byte stream for the testbenches

  --bytes N   bytes of the DDR write/read test, a multiple of 8, default 65536
  --record    with --model: write every byte sent and received to the program
              directory as host_stream_ddrcheck_tx.hex, _rx.hex and _packets.txt
              (the format of run_image.py --record), and _summary.txt (what the
              RUNs of steps 5 and 6 move, as run_image.py --run --record writes
              it), for tb_nexys_top_ddr and tb_dma_ip, which play them to the RTL

Steps, each printed with PASS or FAIL and the first mismatch index:
  1  STATUS: the ddr flag (LED 4: the MIG has calibrated the DDR2) is set
  2  WRITE_DDR N pseudo-random bytes at DDR byte 8 MiB, READ_DDR them back
  3  DMA_LOAD into pool buffers: one item, 37 items from an offset that is
     8- but not 64-byte aligned, 2048 items across 4 KiB pages into buffer 1,
     then READ_BUFFER each region back
  4  DMA_LOAD into the bias store, the LayerNorm constants and the descriptor
     memory; these cannot be read back, so only the response is checked (a DMA
     error or timeout answers with the memory status)
  5  RUN of a run list in DDR with three LOAD entries into pool buffers, then
     READ_BUFFER each region back; the RUN must answer 0 busy cycles (no START)
  6  RUN of a list whose second entry has an unknown kind: the RUN must be
     refused with the run status, after doing the first entry (read back);
     CLEAR then resets the error flag
The engine's buffers are overwritten; run run_image.py afterwards as usual.
Needs numpy; pyserial for a board.
"""

import argparse
import os
import sys
import time

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HOST_DIR)
sys.path.insert(0, HOST_DIR)
import ddr_image  # noqa: E402
import protocol  # noqa: E402

TEST_OFFSET = 8 << 20       # DDR byte offset of step 2, clear of ddr_image's region (1 MiB .. ~2 MiB)
DMA_OFFSET = 16 << 20       # DDR byte offset of the DMA test data
RUN_OFFSET = 24 << 20       # DDR byte offset of the run lists and their data


def compare(name, got, want):
    got, want = np.frombuffer(got, dtype=np.uint8), np.frombuffer(want, dtype=np.uint8)
    if len(got) != len(want):
        print(f"  {name:44s} FAIL  {len(got)} bytes back, {len(want)} expected")
        return False
    wrong = got != want
    if wrong.any():
        print(f"  {name:44s} FAIL  {int(wrong.sum())} of {len(want)} bytes wrong, "
              f"first mismatch index {int(np.argmax(wrong))}")
        return False
    print(f"  {name:44s} PASS  {len(want)} bytes")
    return True


def check(link, total_bytes, rng, run_entries):
    """run_entries collects the run list entries the RUNs carry out (for the summary of --record)."""
    passed = True
    link.clear()
    flags, _cycles = link.status()
    ready = bool(flags & protocol.FLAG_MEMORY)
    print(f"  {'1 DDR calibrated (STATUS ddr flag, LED 4)':44s} {'PASS' if ready else 'FAIL'}")
    if not ready:
        print("  the bitstream must be nexys_top_ddr; the MIG has not calibrated (press CPU_RESET, check the build)")
        return False

    data = rng.integers(0, 256, total_bytes, dtype=np.uint8).tobytes()
    started = time.time()
    link.write_ddr(TEST_OFFSET, data)
    written = time.time()
    back = link.read_ddr(TEST_OFFSET, total_bytes)
    done = time.time()
    passed &= compare("2 WRITE_DDR, READ_DDR", back, data)
    if written > started and done > written:
        print(f"    write {total_bytes / (written - started) / 1e3:.1f} KB/s, "
              f"read {total_bytes / (done - written) / 1e3:.1f} KB/s (the serial link's speed)")

    # step 3: pool buffers, each item is 8 bytes, one per bank
    transfers = [(0, 5, 8, 1), (0, 100, 8 + 64 * 3 + 8, 37), (1, 0, 4096 - 8 * 16, 2048)]
    for buffer_id, address, ddr_shift, items in transfers:
        payload = rng.integers(0, 256, 8 * items, dtype=np.uint8).tobytes()
        link.write_ddr(DMA_OFFSET + ddr_shift, payload)
        link.dma_load(protocol.KIND_POOL, buffer_id, address, DMA_OFFSET + ddr_shift, len(payload))
        back = link.read_buffer(buffer_id, address, items)
        passed &= compare(f"3 DMA_LOAD {items} items to buffer {buffer_id} @ {address}", back, payload)

    # step 4: the memories the host cannot read back
    others = [(protocol.KIND_BIAS, 0, 3, 32 * 5), (protocol.KIND_LAYERNORM, 0, 10, 8 * 12),
              (protocol.KIND_DESCRIPTOR, 0, 200, 24 * 2)]
    for kind, buffer_id, address, length in others:
        payload = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
        if kind == protocol.KIND_DESCRIPTOR:
            # opcode 15 in every entry: a descriptor the sequencer skips, never runs
            payload = b"".join(payload[i:i + 18] + bytes([payload[i + 18], payload[i + 19] | 0x01]) + bytes(4)
                               for i in range(0, length, 24))
        link.write_ddr(DMA_OFFSET + 65536, payload)
        name = {protocol.KIND_BIAS: "bias store", protocol.KIND_LAYERNORM: "LayerNorm constants",
                protocol.KIND_DESCRIPTOR: "descriptor memory"}[kind]
        try:
            link.dma_load(kind, buffer_id, address, DMA_OFFSET + 65536, length)
            print(f"  {'4 DMA_LOAD ' + str(length) + ' bytes to the ' + name:44s} PASS  answered OK")
        except protocol.ProtocolError as error:
            print(f"  {'4 DMA_LOAD ' + str(length) + ' bytes to the ' + name:44s} FAIL  {error}")
            passed = False
    # step 5: a run list of three LOADs; the bridge fetches and runs them by itself
    transfers = [(2, 10, 16), (3, 0, 300), (0, 1000, 7)]
    entries, payloads, data_offset = [], [], RUN_OFFSET + 4096
    for buffer_id, address, items in transfers:
        payload = rng.integers(0, 256, 8 * items, dtype=np.uint8).tobytes()
        link.write_ddr(data_offset, payload)
        entries.append(protocol.load_entry(protocol.KIND_POOL, buffer_id, address, data_offset, len(payload)))
        payloads.append(payload)
        data_offset += -(-len(payload) // 64) * 64
    link.write_ddr(RUN_OFFSET, b"".join(entries))
    try:
        cycles = link.run(RUN_OFFSET, len(entries))
        run_entries += entries
        print(f"  {'5 RUN of 3 LOAD entries':44s} {'PASS' if cycles == 0 else 'FAIL'}  answered, "
              f"{cycles} busy cycles (0 expected)")
        passed &= cycles == 0
    except protocol.ProtocolError as error:
        print(f"  {'5 RUN of 3 LOAD entries':44s} FAIL  {error}")
        passed = False
    for (buffer_id, address, items), payload in zip(transfers, payloads):
        back = link.read_buffer(buffer_id, address, items)
        passed &= compare(f"5   RUN loaded {items} items to buffer {buffer_id} @ {address}", back, payload)

    # step 6: a list whose second entry has kind 7: the first runs, then the RUN stops with the run status
    payload = rng.integers(0, 256, 8 * 5, dtype=np.uint8).tobytes()
    link.write_ddr(data_offset, payload)
    bad = (7).to_bytes(4, "little") + bytes(12)
    entries = [protocol.load_entry(protocol.KIND_POOL, 4, 20, data_offset, len(payload)), bad]
    link.write_ddr(RUN_OFFSET + 1024, b"".join(entries))
    try:
        link.run(RUN_OFFSET + 1024, len(entries))
        print(f"  {'6 RUN with a bad entry is refused':44s} FAIL  the RUN was answered as done")
        passed = False
    except protocol.ProtocolError as error:
        refused = error.status == protocol.STATUS_RUN
        print(f"  {'6 RUN with a bad entry is refused':44s} {'PASS' if refused else 'FAIL'}  status {error.status} "
              f"({protocol.STATUS_RUN} expected)")
        passed &= refused
    run_entries.append(entries[0])
    passed &= compare("6   the entry before it was done", link.read_buffer(4, 20, 5), payload)
    link.clear()

    flags, _cycles = link.status()
    if flags & protocol.FLAG_ERROR:
        print("  the board's error flag is set after the checks: FAIL")
        passed = False
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--bytes", type=int, default=65536)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--program", default=os.path.join(HOST_DIR, "vectors", "mnist112"))
    args = parser.parse_args()
    if (args.port is None) == (not args.model):
        parser.error("give --port or --model")
    if args.bytes <= 0 or args.bytes % 8:
        parser.error("--bytes must be a positive multiple of 8")

    if args.model:
        import engine_model
        from run_image import Program
        program = Program(args.program)
        link = protocol.HostLink(engine_model.ModelDevice(program.depths, os.path.join(
            REPO_DIR, program.config["rom_dir"]), memory=True))
    else:
        import transports
        link = protocol.HostLink(transports.SerialTransport(args.port, args.baud))
    if args.record:
        link.record = []

    print("ddr_check: external memory and DMA")
    run_entries = []
    try:
        passed = check(link, args.bytes, np.random.default_rng(2024), run_entries)
    except protocol.ProtocolError as error:
        print(f"  FAIL: {error}")
        passed = False

    if args.record:
        prefix = os.path.join(args.program, "host_stream_ddrcheck_")
        with open(prefix + "tx.hex", "w") as tx, open(prefix + "rx.hex", "w") as rx, \
                open(prefix + "packets.txt", "w") as packets:
            for packet, response in link.record:
                tx.write("".join(f"{byte:02x}\n" for byte in packet))
                rx.write("".join(f"{byte:02x}\n" for byte in response))
                packets.write(f"{len(packet)} {len(response)} {int(packet[1] == protocol.START)}\n")
        with open(prefix + "summary.txt", "w") as summary:
            summary.write(" ".join(str(value) for value in ddr_image.run_totals(run_entries)) + "\n")
        print(f"recorded {len(link.record)} packets to {prefix}*")
    print("PASS: the DDR, the DMA and RUN work" if passed else "FAIL: see the steps above")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
