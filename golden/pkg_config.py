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
