"""A program in external memory: where each load lives in DDR, the upload, the DMA_LOADs, the run list.

The loads of program_loads.txt (weights, biases, LayerNorm constants) and the
descriptor table are put in DDR once, as the stream items the AXI wrapper takes
(docs/AXI_REGISTER_MAP.md): a pool address is 8 bytes, a bias address 32, a
LayerNorm pair 8 (gain, offset, two zero bytes), a descriptor 24 (20 bytes, four
zero bytes). Each load starts on a 64-byte boundary from IMAGE_OFFSET on, so the
DMA needs no unaligned transfers. The input image changes per run: the
step-by-step flow writes it with WRITE_BUFFER, the RUN flow with WRITE_DDR to
INPUT_OFFSET.

The run list (after the loads, also part of the image) is what RUN works
through: LOAD the descriptors, then per phase a LOAD per load of
program_loads.txt (the input image from INPUT_OFFSET) and a START. A program
longer than the descriptor memory gets its descriptors phase by phase instead
(run_image.Program.descriptor_blocks), each before its phase's START.

A 64-byte signature at SIGNATURE_OFFSET (a magic word, the SHA-256 of the
image and its length) says which image the DDR holds, so a second run on the
same powered board reads 64 bytes instead of uploading the image again.
"""

import hashlib

import protocol

IMAGE_OFFSET = 1 << 20                  # first byte of the loads in DDR
SIGNATURE_OFFSET = IMAGE_OFFSET - 64
INPUT_OFFSET = 0                        # the input image of the RUN flow, below the signature
DESCRIPTOR_FILE = "program_desc.hex"
ALIGN = 64
MAGIC = b"VIT-DDR1"
PER_IMAGE_LOADS = ("embed_in_image.hex",)
KINDS = {"buffer": protocol.KIND_POOL, "bias": protocol.KIND_BIAS, "layernorm": protocol.KIND_LAYERNORM}


class DdrImage:
    def __init__(self, program):
        """program: run_image.Program. placements: load file -> (kind, DDR offset, bytes)."""
        self.placements = {}
        image = bytearray()

        def place(file_name, kind, data):
            nonlocal image
            image += bytes(-len(image) % ALIGN)
            self.placements[file_name] = (kind, IMAGE_OFFSET + len(image), len(data))
            image += data

        self.input_bytes = 0
        for _phase, kind, _target, _base, words, file_name in program.loads:
            if file_name in PER_IMAGE_LOADS:
                self.input_bytes = 8 * words
            if file_name in PER_IMAGE_LOADS or file_name in self.placements:
                continue
            data = program.load_bytes(kind, file_name)
            if kind == "layernorm":     # 6-byte host items, 8-byte stream beats
                data = b"".join(data[i:i + 6] + b"\0\0" for i in range(0, len(data), 6))
            place(file_name, KINDS[kind], data)
        descriptors = program.descriptor_bytes
        place(DESCRIPTOR_FILE, protocol.KIND_DESCRIPTOR,
              b"".join(descriptors[i:i + 20] + bytes(4) for i in range(0, len(descriptors), 20)))

        # the run list
        descriptor_kind, descriptor_offset, _length = self.placements[DESCRIPTOR_FILE]
        blocks = program.descriptor_blocks()

        def descriptors(phase):
            return [protocol.load_entry(descriptor_kind, 0, first % protocol.DESCRIPTOR_SLOTS,
                                        descriptor_offset + 24 * first, 24 * count)
                    for block_phase, first, count in blocks if block_phase == phase]

        entries = descriptors(None)
        for phase, first, _count in program.phases:
            entries += descriptors(phase)
            for load_phase, load_kind, target, base, _words, file_name in program.loads:
                if load_phase != phase:
                    continue
                if file_name in PER_IMAGE_LOADS:
                    kind, offset, length = KINDS[load_kind], INPUT_OFFSET, self.input_bytes
                else:
                    kind, offset, length = self.placements[file_name]
                entries.append(protocol.load_entry(kind, target, base, offset, length))
            entries.append(protocol.start_entry(first % protocol.DESCRIPTOR_SLOTS))
        self.entries = entries
        image += bytes(-len(image) % ALIGN)
        self.list_offset = IMAGE_OFFSET + len(image)
        image += b"".join(entries)
        image += bytes(-len(image) % ALIGN)
        self.image = bytes(image)
        assert INPUT_OFFSET + self.input_bytes <= SIGNATURE_OFFSET
        self.signature = (MAGIC + hashlib.sha256(self.image).digest() + len(self.image).to_bytes(8, "little")
                          + bytes(16))
        assert len(self.signature) == 64 and IMAGE_OFFSET + len(self.image) <= protocol.DDR_BYTES

    def upload(self, link, reload=False):
        """Writes the image unless the DDR already holds it; True if it was written."""
        if not reload and link.read_ddr(SIGNATURE_OFFSET, len(self.signature)) == self.signature:
            return False
        link.write_ddr(IMAGE_OFFSET, self.image)
        link.write_ddr(SIGNATURE_OFFSET, self.signature)
        return True

    def load(self, link, file_name, target, base):
        """The DMA_LOAD of one load of program_loads.txt."""
        kind, offset, length = self.placements[file_name]
        link.dma_load(kind, target, base, offset, length)

    def run(self, link, input_data):
        """The RUN flow: the input image to INPUT_OFFSET, then one RUN of the whole list. Returns the busy cycles."""
        assert len(input_data) == self.input_bytes
        link.write_ddr(INPUT_OFFSET, input_data)
        return link.run(self.list_offset, len(self.entries))

    def run_totals(self):
        """What one RUN of the whole list moves, for tb_axi_replay_run_mnist112."""
        return run_totals(self.entries)


def run_totals(entries):
    """What RUN moves for these entries: (LOAD entries, DMA beats, pool items, bias items, LayerNorm items,
    descriptor words, START entries), the numbers the replay testbenches check."""
    loads = beats = pool = bias = layernorm = descriptor_words = starts = 0
    for entry in entries:
        word0, length = int.from_bytes(entry[0:4], "little"), int.from_bytes(entry[12:16], "little")
        if word0 & 0xFF == protocol.ENTRY_START:
            starts += 1
            continue
        kind = word0 >> 12 & 3
        loads, beats = loads + 1, beats + length // 8
        if kind == protocol.KIND_POOL:
            pool += length // 8
        elif kind == protocol.KIND_BIAS:
            bias += length // 32
        elif kind == protocol.KIND_LAYERNORM:
            layernorm += length // 8
        else:
            descriptor_words += 5 * (length // 24)
    return loads, beats, pool, bias, layernorm, descriptor_words, starts
