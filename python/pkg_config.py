"""Reads rtl/vit_config_pkg.sv so the golden model and the RTL share one definition.

PARAMS holds every package parameter as an int. DESCRIPTOR_FIELDS lists the
fields of descriptor_t as (name, width, signed), most significant first, the
same order as the packed struct.
"""

import os
import re

PACKAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "rtl", "vit_config_pkg.sv")


def _clog2(value):
    return (value - 1).bit_length()


def _evaluate(expr, params, struct_widths):
    expr = re.sub(r"\$bits\((\w+)\)", lambda m: str(struct_widths[m.group(1)]), expr)
    expr = re.sub(r"\d+'([bdh])([0-9a-fA-F]+)",
                  lambda m: str(int(m.group(2), {"b": 2, "d": 10, "h": 16}[m.group(1)])), expr)
    expr = expr.replace("$clog2", "clog2").replace("/", "//")
    return int(eval(expr, {"__builtins__": {}}, {**params, "clog2": _clog2}))


def _load():
    with open(PACKAGE_FILE) as f:
        text = re.sub(r"//[^\n]*", "", f.read())
    params, structs, struct_widths = {}, {}, {}
    statement = re.compile(
        r"parameter\s+(?P<name>\w+)\s*=\s*(?P<expr>[^;]+);"
        r"|typedef\s+struct\s+packed\s*\{(?P<body>.*?)\}\s*(?P<struct>\w+)\s*;", re.S)
    field = re.compile(r"logic\s*(signed)?\s*(?:\[([^\]]+):0\])?\s*(\w+)\s*;")
    for m in statement.finditer(text):
        if m.group("name"):
            params[m.group("name")] = _evaluate(m.group("expr"), params, struct_widths)
        else:
            fields = [(name, _evaluate(msb, params, struct_widths) + 1 if msb else 1, bool(signed))
                      for signed, msb, name in field.findall(m.group("body"))]
            structs[m.group("struct")] = fields
            struct_widths[m.group("struct")] = sum(width for _, width, _ in fields)
    return params, structs


PARAMS, _STRUCTS = _load()
DESCRIPTOR_FIELDS = _STRUCTS["descriptor_t"]

assert PARAMS["DESCRIPTOR_WIDTH"] == sum(width for _, width, _ in DESCRIPTOR_FIELDS)


def pack_descriptor(**values):
    """Pack one descriptor into DESCRIPTOR_WORDS 32-bit words, word 0 least significant."""
    names = [name for name, _, _ in DESCRIPTOR_FIELDS]
    if set(values) != set(names):
        raise ValueError(f"descriptor fields differ from the package: {sorted(set(values) ^ set(names))}")
    packed = 0
    for name, width, signed in DESCRIPTOR_FIELDS:
        value = int(values[name])
        lo, hi = (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)
        if not lo <= value <= hi:
            raise ValueError(f"descriptor field {name}={value} does not fit {width} bits")
        packed = (packed << width) | (value & ((1 << width) - 1))
    return [(packed >> (32 * i)) & 0xFFFFFFFF for i in range(PARAMS["DESCRIPTOR_WORDS"])]


def unpack_descriptor(words):
    """The inverse of pack_descriptor: 32-bit words, word 0 least significant, to a dict."""
    packed = sum(int(word) << (32 * i) for i, word in enumerate(words))
    values, position = {}, PARAMS["DESCRIPTOR_WIDTH"]
    for name, width, signed in DESCRIPTOR_FIELDS:
        position -= width
        value = (packed >> position) & ((1 << width) - 1)
        if signed and value >> (width - 1):
            value -= 1 << width
        values[name] = value
    return values


# what each descriptor field means, for the table in docs/ENGINE.md
DESCRIPTOR_FIELD_NOTES = {
    "opcode": "Operation. 0 = MATMUL, 1 = ADD, 2 = GELU, 3 = LAYERNORM, 4 = SOFTMAX; any other value is skipped.",
    "rows_total": "M, rows of the activation operand and of the result.",
    "inner_dim": "K, the dimension summed over. At least 1.",
    "cols_total": "N, columns of the weight operand and of the result.",
    "row_tiles": "ceil(M / SYS_ROW), worked out by the host.",
    "col_tiles": "ceil(N / SYS_COL), worked out by the host.",
    "activation_buffer_id": "Pool buffer read as the activation operand (row banks).",
    "weight_buffer_id": "Pool buffer read as the weight operand (column banks).",
    "output_buffer_id": "Pool buffer the result is written to.",
    "activation_base": "Address of the activation operand's first word in each bank.",
    "weight_base": "Address of the weight operand's first word in each bank.",
    "output_base": "Address of the result's first word in each bank.",
    "output_stride": "Words per stored row of the result. N for column-select and M for "
                     "row-select, or more to place the result inside a wider tensor.",
    "bias_base": "Address in the bias store of the bias words for output columns 0 .. SYS_COL-1.",
    "scale": "Requantizer multiplier, signed.",
    "shift": "Requantizer right shift, arithmetic.",
    "drain_mode": "0 = row-select (result in column banks), 1 = column-select (result in row banks).",
    "bias_enable": "1 = add the bias before requantizing.",
    "last_op": "1 = stop after this op and pulse done.",
}


def descriptor_markdown():
    """The descriptor_t layout as a Markdown table, most significant field first."""
    lines = ["| Field | Bits | Width | Meaning |", "|---|---|---|---|"]
    position = PARAMS["DESCRIPTOR_WIDTH"]
    for name, width, signed in DESCRIPTOR_FIELDS:
        position -= width
        bits = f"{position + width - 1}:{position}" if width > 1 else f"{position}"
        kind = f"{width}, signed" if signed else f"{width}"
        lines.append(f"| `{name}` | {bits} | {kind} | {DESCRIPTOR_FIELD_NOTES[name]} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(descriptor_markdown())
    print(f"\n{PARAMS['DESCRIPTOR_WIDTH']} bits, stored in {PARAMS['DESCRIPTOR_WORDS']} 32-bit words "
          f"({PARAMS['DESCRIPTOR_MEMORY_WIDTH']} bits).")
