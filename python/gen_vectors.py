"""Generate golden .hex vectors for the matmul (systolic array) and requantizer.

Usage: python3 python/gen_vectors.py
Output: python/vectors/*.hex, all in $readmemh format (one word per line,
two's complement, matrices in row-major order).

Matmul cases, <case> = mm_small, mm_ragged, mm_extreme, mm_qkv, mm_k1, mm_k2,
mm_nobias:
  <case>_cfg.hex    32-bit words: M, K, N, scale, shift
  <case>_A.hex      M*K  INT8   A[m][k]
  <case>_B.hex      K*N  INT8   B[k][n]
  <case>_bias.hex   N    INT32  bias[n], one per output column
  <case>_C_acc.hex  M*N  INT32  A @ B: accumulators held in the PEs, no bias
  <case>_C_bias.hex M*N  INT32  A @ B + bias[n]: requantizer input
  <case>_Y.hex      M*N  INT8   requantized output (of C_acc for mm_nobias)
  <case>_desc_r<R>c<C>_<row|col>.hex
                    one-op descriptor table for matmul_engine on an R x C
                    array, row-select or column-select drain: descriptor_t
                    from rtl/core/vit_config_pkg.sv as 32-bit words, word 0 least
                    significant

Chained attention head for tb_matmul_engine_chain, DeiT-Tiny dimensions
(197 tokens, embed 192, head dimension 64), softmax left out:
  Q = rq(X @ Wq + bq), K = rq(X @ Wk + bk), V = rq(X @ Wv + bv)
  S = rq(Q @ K^T), O = rq(S @ V)                 (rq = requantize, no bias)
  attn_cfg.hex    32-bit words: tokens, embed, head dimension, weight columns
  attn_X.hex      197*192  INT8   X[token][k]
  attn_W.hex      192*576  INT8   qkv weight W[k][col], columns q | k | v
  attn_bias.hex   576      INT32  qkv bias[col]
  attn_Q.hex, attn_K.hex, attn_V.hex  197*64   INT8
  attn_S.hex      197*197  INT8   0..127
  attn_O.hex      197*64   INT8
  attn_desc_r<R>c<C>.hex   five-op descriptor table (Q, K, V, S, O)

Single output tile for tb_systolic_array, <case> = tile_r<R>c<C>_k<K> with
R x C in 2x2, 4x4, 3x5, 12x12, 16x16 and K in 1, 3, 8, 197:
  <case>_A.hex      R*K  INT8   A[row][k]
  <case>_B.hex      K*C  INT8   B[k][col]
  <case>_C_acc.hex  R*C  INT32  A @ B, no bias

Requantizer on its own, vector i is one (acc, scale, shift) -> out:
  requant_cfg.hex    32-bit word: number of vectors
  requant_acc.hex    INT32
  requant_scale.hex  INT8 (signed)
  requant_shift.hex  8-bit word holding the 5-bit shift
  requant_out.hex    INT8
"""

import os

import numpy as np

from pkg_config import PARAMS, pack_descriptor
from int8_ops import (ACC_MAX, ACC_MIN, ACC_WIDTH, DATA_MAX, DATA_MIN, DATA_WIDTH,
                      SCALE_MAX, SCALE_MIN, SCALE_WIDTH, SHIFT_MAX, add_bias,
                      linear_acc, matmul_acc, read_hex, requantize, wrap_signed,
                      write_hex)

VEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")
CFG_WIDTH = 32

# array sizes the engine descriptor tables are generated for, and the bank
# depth tb_matmul_engine gives every pool buffer
ENGINE_ARRAYS = ((4, 4), (8, 8), (12, 12), (16, 16))
ENGINE_TB_BANK_DEPTH = 16384


def pick_scale_shift(acc, target_peak=160):
    """Pick (scale, shift) so the largest |acc| lands near target_peak.

    A peak above 127 makes a few outputs saturate, which exercises the clip.
    Integer-only so the choice is reproducible everywhere.
    """
    max_abs = int(np.abs(acc).max())
    if max_abs == 0:
        return 1, 0
    for shift in range(SHIFT_MAX + 1):
        scale = ((target_peak << shift) + max_abs // 2) // max_abs
        if scale >= 64:
            break
    return min(scale, SCALE_MAX), shift


def ref_matmul(a, b):
    """Slow reference: one PE accumulator per output, wrapped every cycle."""
    m_dim, k_dim = a.shape
    n_dim = b.shape[1]
    c = np.zeros((m_dim, n_dim), dtype=np.int64)
    for m in range(m_dim):
        for n in range(n_dim):
            acc = 0
            for k in range(k_dim):
                acc = int(wrap_signed(acc + int(a[m, k]) * int(b[k, n]), ACC_WIDTH))
            c[m, n] = acc
    return c


def ref_requantize(acc, scale, shift):
    """Slow reference on Python ints."""
    return max(DATA_MIN, min(DATA_MAX, (int(acc) * int(scale)) >> int(shift)))


def path(name):
    return os.path.join(VEC_DIR, name)


def rand_bias(rng, a, b):
    """Random bias[n] up to a quarter of the largest |A @ B| accumulator."""
    lim = max(1, int(np.abs(matmul_acc(a, b)).max()) // 4)
    return rng.integers(-lim, lim + 1, size=b.shape[1], dtype=np.int64)


def dump_descriptors(case, m_dim, k_dim, n_dim, scale, shift, bias_enable, offset_layout):
    """One-op engine tables for a matmul case, per array size and drain mode.

    The two drain modes use different buffer ids. offset_layout puts every
    operand at a nonzero base and pads the output stride, so the result lands
    inside a wider stored tensor.
    """
    for sys_row, sys_col in ENGINE_ARRAYS:
        row_tiles = -(-m_dim // sys_row)
        col_tiles = -(-n_dim // sys_col)
        for mode_name, mode, ids in (("row", PARAMS["DRAIN_ROW_SELECT"], (0, 1, 2)),
                                     ("col", PARAMS["DRAIN_COLUMN_SELECT"], (3, 5, 6))):
            column_select = mode == PARAMS["DRAIN_COLUMN_SELECT"]
            bank_rows = row_tiles if column_select else col_tiles
            stride = (n_dim if column_select else m_dim) + (2 if offset_layout else 0)
            fields = dict(
                opcode=PARAMS["OPCODE_MATMUL"],
                rows_total=m_dim, inner_dim=k_dim, cols_total=n_dim,
                row_tiles=row_tiles, col_tiles=col_tiles,
                activation_buffer_id=ids[0], weight_buffer_id=ids[1], output_buffer_id=ids[2],
                activation_base=5 if offset_layout else 0,
                weight_base=9 if offset_layout else 0,
                output_base=11 if offset_layout else 0,
                output_stride=stride,
                bias_base=3 if offset_layout else 0,
                scale=scale, shift=shift, drain_mode=mode,
                bias_enable=int(bias_enable), last_op=1)
            assert fields["activation_base"] + row_tiles * k_dim <= ENGINE_TB_BANK_DEPTH, case
            assert fields["weight_base"] + col_tiles * k_dim <= ENGINE_TB_BANK_DEPTH, case
            assert fields["output_base"] + bank_rows * stride <= ENGINE_TB_BANK_DEPTH, case
            assert fields["bias_base"] + col_tiles <= PARAMS["BIAS_DEPTH"], case
            write_hex(path(f"{case}_desc_r{sys_row}c{sys_col}_{mode_name}.hex"),
                      pack_descriptor(**fields), 32)


def dump_matmul(case, a, b, bias, bias_enable=True, offset_layout=False):
    c = matmul_acc(a, b)
    # scale/shift come from the accumulators alone, so adding a bias to a case
    # leaves its cfg unchanged
    scale, shift = pick_scale_shift(c)
    cb = linear_acc(a, b, bias)
    y = requantize(cb if bias_enable else c, scale, shift)

    # cross-check the vectorised model against the slow per-PE reference
    if a.shape[0] * a.shape[1] * b.shape[1] <= 100_000:
        assert np.array_equal(c, ref_matmul(a, b)), f"{case}: matmul model mismatch"
    cb_ref = [[int(wrap_signed(int(c[m, n]) + int(bias[n]), ACC_WIDTH))
               for n in range(c.shape[1])] for m in range(c.shape[0])]
    assert np.array_equal(cb, cb_ref), f"{case}: bias model mismatch"
    assert np.array_equal(cb, add_bias(c, bias)), f"{case}: bias model mismatch"
    y_ref = [ref_requantize(v, scale, shift) for v in (cb if bias_enable else c).reshape(-1)]
    assert np.array_equal(y.reshape(-1), y_ref), f"{case}: requant model mismatch"

    m_dim, k_dim = a.shape
    n_dim = b.shape[1]
    write_hex(path(f"{case}_cfg.hex"), [m_dim, k_dim, n_dim, scale, shift], CFG_WIDTH)
    write_hex(path(f"{case}_A.hex"), a, DATA_WIDTH)
    write_hex(path(f"{case}_B.hex"), b, DATA_WIDTH)
    write_hex(path(f"{case}_bias.hex"), bias, ACC_WIDTH)
    write_hex(path(f"{case}_C_acc.hex"), c, ACC_WIDTH)
    write_hex(path(f"{case}_C_bias.hex"), cb, ACC_WIDTH)
    write_hex(path(f"{case}_Y.hex"), y, DATA_WIDTH)
    dump_descriptors(case, m_dim, k_dim, n_dim, scale, shift, bias_enable, offset_layout)

    # make sure what landed on disk is what the model produced
    assert np.array_equal(read_hex(path(f"{case}_bias.hex"), ACC_WIDTH), bias)
    assert np.array_equal(read_hex(path(f"{case}_C_acc.hex"), ACC_WIDTH), c.reshape(-1))
    assert np.array_equal(read_hex(path(f"{case}_C_bias.hex"), ACC_WIDTH), cb.reshape(-1))
    assert np.array_equal(read_hex(path(f"{case}_Y.hex"), DATA_WIDTH), y.reshape(-1))

    sat = int(np.count_nonzero((y == DATA_MIN) | (y == DATA_MAX)))
    print(f"{case:11s} {m_dim}x{k_dim} @ {k_dim}x{n_dim}  scale={scale} shift={shift}  "
          f"acc+bias range [{cb.min()}, {cb.max()}]  saturated {sat}/{y.size}")


def rand_int8(rng, shape):
    return rng.integers(DATA_MIN, DATA_MAX + 1, size=shape, dtype=np.int64)


def gen_matmul():
    # bias draws use their own generators so the A and B data never shift

    # one 8x8 output tile, K longer than the tile
    rng = np.random.default_rng(1)
    a, b = rand_int8(rng, (8, 16)), rand_int8(rng, (16, 8))
    dump_matmul("mm_small", a, b, rand_bias(np.random.default_rng(11), a, b))

    # M and N not multiples of any likely tile size: edge tiles are partial
    rng = np.random.default_rng(2)
    a, b = rand_int8(rng, (13, 24)), rand_int8(rng, (24, 11))
    dump_matmul("mm_ragged", a, b, rand_bias(np.random.default_rng(12), a, b), offset_layout=True)

    # worst-case accumulation over the longest K in the model (MLP hidden 768);
    # the extreme biases push columns 0 and 1 past INT32 so the drain adder wraps
    rng = np.random.default_rng(3)
    k_dim = 768
    alt = np.where(np.arange(k_dim) % 2 == 0, DATA_MIN, DATA_MAX)
    a = np.stack([np.full(k_dim, DATA_MIN), np.full(k_dim, DATA_MAX), alt,
                  rand_int8(rng, k_dim)])
    b = np.stack([np.full(k_dim, DATA_MIN), np.full(k_dim, DATA_MAX), alt,
                  rand_int8(rng, k_dim)], axis=1)
    bias = np.array([ACC_MAX, ACC_MIN, 0, int(rand_bias(np.random.default_rng(13), a, b)[3])])
    dump_matmul("mm_extreme", a, b, bias)

    # real layer shape: 197 tokens x embed 192 through a 192x192 linear
    rng = np.random.default_rng(4)
    a, b = rand_int8(rng, (197, 192)), rand_int8(rng, (192, 192))
    dump_matmul("mm_qkv", a, b, rand_bias(np.random.default_rng(14), a, b))


def gen_short_k():
    # K = 1 and K = 2: the shortest tiles, so consecutive tiles are as close
    # together as they can get; ragged in M and N at both 4x4 and 8x8
    for k_dim in (1, 2):
        rng = np.random.default_rng([30, k_dim])
        a, b = rand_int8(rng, (19, k_dim)), rand_int8(rng, (k_dim, 21))
        dump_matmul(f"mm_k{k_dim}", a, b, rand_bias(np.random.default_rng([31, k_dim]), a, b),
                    offset_layout=True)

    # bias disabled: Y is the requantized A @ B, the stored bias must be ignored
    rng = np.random.default_rng(32)
    a, b = rand_int8(rng, (11, 5)), rand_int8(rng, (5, 9))
    dump_matmul("mm_nobias", a, b, rand_bias(np.random.default_rng(33), a, b),
                bias_enable=False, offset_layout=True)


def gen_attention_chain():
    tokens, embed, head_dim = 197, 192, 64
    head = 0
    q_col, k_col, v_col = head * head_dim, embed + head * head_dim, 2 * embed + head * head_dim

    rng = np.random.default_rng(40)
    x = rand_int8(rng, (tokens, embed))
    w = rand_int8(rng, (embed, 3 * embed))

    # The q and k biases lift every accumulator to zero or above, so Q and K
    # are 0..127 and S = Q @ K^T cannot go negative: S then stands in for the
    # softmax output P. The v bias is an ordinary signed one.
    acc_full = matmul_acc(x, w)
    bias = np.empty(3 * embed, dtype=np.int64)
    bias[:2 * embed] = -acc_full[:, :2 * embed].min(axis=0) + rng.integers(0, 2000, size=2 * embed)
    v_lim = int(np.abs(acc_full[:, 2 * embed:]).max()) // 4
    bias[2 * embed:] = rng.integers(-v_lim, v_lim + 1, size=embed)

    def linear(col, target_peak):
        acc = linear_acc(x, w[:, col:col + head_dim], bias[col:col + head_dim])
        scale, shift = pick_scale_shift(acc, target_peak)
        return requantize(acc, scale, shift), scale, shift

    q, q_scale, q_shift = linear(q_col, 127)
    k, k_scale, k_shift = linear(k_col, 127)
    v, v_scale, v_shift = linear(v_col, 160)
    assert q.min() >= 0 and k.min() >= 0

    # attention scores: the 1/sqrt(head_dim) = 1/8 is three extra shift bits
    s_acc = matmul_acc(q, k.T)
    s_scale, s_shift = pick_scale_shift(s_acc >> 3, 127)
    s_shift += 3
    assert s_shift <= SHIFT_MAX
    s = requantize(s_acc, s_scale, s_shift)
    assert s.min() >= 0 and s.max() <= DATA_MAX

    o_acc = matmul_acc(s, v)
    o_scale, o_shift = pick_scale_shift(o_acc)
    o = requantize(o_acc, o_scale, o_shift)

    write_hex(path("attn_cfg.hex"), [tokens, embed, head_dim, 3 * embed], CFG_WIDTH)
    write_hex(path("attn_X.hex"), x, DATA_WIDTH)
    write_hex(path("attn_W.hex"), w, DATA_WIDTH)
    write_hex(path("attn_bias.hex"), bias, ACC_WIDTH)
    for name, tensor in (("Q", q), ("K", k), ("V", v), ("S", s), ("O", o)):
        write_hex(path(f"attn_{name}.hex"), tensor, DATA_WIDTH)
        assert np.array_equal(read_hex(path(f"attn_{name}.hex"), DATA_WIDTH), tensor.reshape(-1))

    # buffer ids: X 0, qkv weight 1, Q 2, K 3, V 4, S 5, O 6
    row_select, column_select = PARAMS["DRAIN_ROW_SELECT"], PARAMS["DRAIN_COLUMN_SELECT"]
    for sys_row, sys_col in ENGINE_ARRAYS:
        def op(rows, inner, cols, ids, weight_col, stride, scale, shift, mode, bias_enable, last_op=0):
            return pack_descriptor(
                opcode=PARAMS["OPCODE_MATMUL"],
                rows_total=rows, inner_dim=inner, cols_total=cols,
                row_tiles=-(-rows // sys_row), col_tiles=-(-cols // sys_col),
                activation_buffer_id=ids[0], weight_buffer_id=ids[1], output_buffer_id=ids[2],
                activation_base=0,
                # a column slice of the stored weight starts at a whole column tile
                weight_base=(weight_col // sys_col) * inner,
                output_base=0, output_stride=stride,
                bias_base=weight_col // sys_col,
                scale=scale, shift=shift, drain_mode=mode,
                bias_enable=bias_enable, last_op=last_op)

        assert q_col % sys_col == 0 and k_col % sys_col == 0 and v_col % sys_col == 0
        table = (
            op(tokens, embed, head_dim, (0, 1, 2), q_col, head_dim, q_scale, q_shift, column_select, 1)
            + op(tokens, embed, head_dim, (0, 1, 3), k_col, head_dim, k_scale, k_shift, column_select, 1)
            + op(tokens, embed, head_dim, (0, 1, 4), v_col, tokens, v_scale, v_shift, row_select, 1)
            + op(tokens, head_dim, tokens, (2, 3, 5), 0, tokens, s_scale, s_shift, column_select, 0)
            # O goes into the head's slot of a 197 x 192 tensor: stride 192
            + op(tokens, tokens, head_dim, (5, 4, 6), 0, embed, o_scale, o_shift, column_select, 0, last_op=1))
        write_hex(path(f"attn_desc_r{sys_row}c{sys_col}.hex"), table, 32)

    print(f"{'attn':11s} Q [{q.min()}, {q.max()}] scale={q_scale} shift={q_shift}  "
          f"K [{k.min()}, {k.max()}] scale={k_scale} shift={k_shift}  "
          f"V [{v.min()}, {v.max()}] scale={v_scale} shift={v_shift}")
    print(f"{'':11s} S [{s.min()}, {s.max()}] scale={s_scale} shift={s_shift} (3 bits are the 1/8)  "
          f"O [{o.min()}, {o.max()}] scale={o_scale} shift={o_shift}")


def gen_tile():
    count = 0
    for rows, cols in ((2, 2), (4, 4), (3, 5), (12, 12), (16, 16)):
        for k_dim in (1, 3, 8, 197):
            case = f"tile_r{rows}c{cols}_k{k_dim}"
            rng = np.random.default_rng([20, rows, cols, k_dim])
            a, b = rand_int8(rng, (rows, k_dim)), rand_int8(rng, (k_dim, cols))
            c = matmul_acc(a, b)
            assert np.array_equal(c, ref_matmul(a, b)), f"{case}: matmul model mismatch"
            write_hex(path(f"{case}_A.hex"), a, DATA_WIDTH)
            write_hex(path(f"{case}_B.hex"), b, DATA_WIDTH)
            write_hex(path(f"{case}_C_acc.hex"), c, ACC_WIDTH)
            assert np.array_equal(read_hex(path(f"{case}_C_acc.hex"), ACC_WIDTH), c.reshape(-1))
            count += 1
    print(f"{'tile':11s} {count} single-tile cases (2x2, 4x4, 3x5, 12x12, 16x16 arrays; K = 1, 3, 8, 197)")


def gen_requant():
    vecs = []

    # corner values of every input, all combinations
    accs = [0, 1, -1, 127, 128, -128, -129, 255, 256, 1 << 30, -(1 << 30),
            768 * 128 * 128, 768 * -128 * 127, ACC_MAX, ACC_MIN]
    scales = [SCALE_MIN, -127, -1, 0, 1, 2, 64, SCALE_MAX]
    shifts = [0, 1, 2, 7, 8, 15, 16, 30, SHIFT_MAX]
    vecs += [(a, sc, sh) for a in accs for sc in scales for sh in shifts]

    # results that sit right on the clip boundaries
    for sh in (0, 3, 8, 14, 20):
        for sc in (1, 3, 77, SCALE_MAX, -1, -90, SCALE_MIN):
            for target in (126, 127, 128, -127, -128, -129):
                base = (target << sh) // sc
                for a in (base - 1, base, base + 1):
                    if ACC_MIN <= a <= ACC_MAX:
                        vecs.append((a, sc, sh))

    # random, with acc magnitude spread over all bit widths
    rng = np.random.default_rng(5)
    for _ in range(4000):
        bits = int(rng.integers(1, ACC_WIDTH + 1))
        a = int(rng.integers(-(1 << (bits - 1)), 1 << (bits - 1)))
        sc = int(rng.integers(SCALE_MIN, SCALE_MAX + 1))
        sh = int(rng.integers(0, SHIFT_MAX + 1))
        vecs.append((a, sc, sh))

    acc, scale, shift = (np.array(col, dtype=np.int64) for col in zip(*vecs))
    out = requantize(acc, scale, shift)
    assert np.array_equal(out, [ref_requantize(*v) for v in vecs]), "requant model mismatch"

    write_hex(path("requant_cfg.hex"), [len(vecs)], CFG_WIDTH)
    write_hex(path("requant_acc.hex"), acc, ACC_WIDTH)
    write_hex(path("requant_scale.hex"), scale, SCALE_WIDTH)
    write_hex(path("requant_shift.hex"), shift, 8)
    write_hex(path("requant_out.hex"), out, DATA_WIDTH)
    assert np.array_equal(read_hex(path("requant_out.hex"), DATA_WIDTH), out)

    sat = int(np.count_nonzero((out == DATA_MIN) | (out == DATA_MAX)))
    print(f"{'requant':11s} {len(vecs)} vectors  saturated {sat}/{len(vecs)}")


if __name__ == "__main__":
    os.makedirs(VEC_DIR, exist_ok=True)
    gen_matmul()
    gen_short_k()
    gen_attention_chain()
    gen_tile()
    gen_requant()
    print(f"wrote {len(os.listdir(VEC_DIR))} files to {VEC_DIR}")
