"""A program's loads in external memory: where each one lives in DDR, the upload, the DMA_LOADs.

The loads of program_loads.txt (weights, biases, LayerNorm constants) are put in
DDR once, as the stream items the AXI wrapper takes (docs/AXI_REGISTER_MAP.md):
a pool address is 8 bytes, a bias address 32, a LayerNorm pair 8 (gain, offset,
two zero bytes). Each load starts on a 64-byte boundary from IMAGE_OFFSET on, so
the DMA needs no unaligned transfers. The image changes per run is not put in
DDR; the host writes it with WRITE_BUFFER as before.

A 64-byte signature at SIGNATURE_OFFSET (a magic word, the SHA-256 of the
image and its length) says which image the DDR holds, so a second run on the
same powered board reads 64 bytes instead of uploading the image again.
"""

import hashlib

import protocol

IMAGE_OFFSET = 1 << 20                  # first byte of the loads in DDR
SIGNATURE_OFFSET = IMAGE_OFFSET - 64
ALIGN = 64
MAGIC = b"VIT-DDR1"
PER_IMAGE_LOADS = ("embed_in_image.hex",)
KINDS = {"buffer": protocol.KIND_POOL, "bias": protocol.KIND_BIAS, "layernorm": protocol.KIND_LAYERNORM}


class DdrImage:
    def __init__(self, program):
        """program: run_image.Program. placements: load file -> (kind, DDR offset, bytes)."""
        self.placements = {}
        image = bytearray()
        for _phase, kind, _target, _base, _words, file_name in program.loads:
            if file_name in PER_IMAGE_LOADS or file_name in self.placements:
                continue
            data = program.load_bytes(kind, file_name)
            if kind == "layernorm":     # 6-byte host items, 8-byte stream beats
                data = b"".join(data[i:i + 6] + b"\0\0" for i in range(0, len(data), 6))
            image += bytes(-len(image) % ALIGN)
            self.placements[file_name] = (KINDS[kind], IMAGE_OFFSET + len(image), len(data))
            image += data
        image += bytes(-len(image) % ALIGN)
        self.image = bytes(image)
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
