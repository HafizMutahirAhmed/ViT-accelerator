# Schedule

Two fixed dates:

| Date | What |
|---|---|
| Oct 20, 2026 | Hardware bring-up of the matmul engine must have started by this day |
| Nov 2, 2026 | Project deadline |

## Where the project is

As of Oct 12, in simulation only; nothing has run on the board yet.

- **Matmul engine**: frozen at the tag `engine-v1.1`, described in
  [ENGINE.md](ENGINE.md).
- **Non-linear ops**: residual add, GELU, LayerNorm and softmax are row ops of
  the engine (tags `nl-add`, `nl-gelu`, `nl-layernorm`, `nl-softmax`).
- **Full block, loop x2, classifier**: the whole 2-layer pass runs as one
  57-entry descriptor program (tag `block-v1`): patch embedding, class token
  and position add, both encoder layers, the final LayerNorm and both
  classifier heads. `tb_encoder_program` loads everything through the host
  port and finds every one of the 57 stages bit-exact against the golden
  model. One image takes 4,190,681 cycles, 1,844,272 per layer.
- **Build**: 8x8 out of context at 6.667 ns meets timing with WNS +0.058 ns,
  using 326 of 365 block RAM tiles.
- **Golden model**: scales from 128 calibration images; the layer-2 output
  has cosine 0.977 against FP32 on 42 held-out images.

## Order of work

1. **Non-linear ops.** Softmax, LayerNorm, GELU and the residual add, each with
   its own golden vectors and self-checking testbench. Done in simulation.
2. **Full block.** One encoder layer end to end: the engine ops and the
   non-linear ops in sequence, bit-exact against the golden model. Done in
   simulation.
3. **Loop x2.** The same block run twice, for layers 0 and 1, with the weights
   of each layer. Done in simulation.
4. **Classifier.** The linear head on the CLS row and the argmax. The head is
   done in simulation; the argmax is not built.
5. **Integration.** Everything together on the board. Not started: MIG/DDR3,
   the FrontPanel host interface and the top level do not exist yet.

Hardware bring-up of the engine is not a step in this list. It runs alongside
it and has to start by Oct 20, whichever step is in progress then.

[SPEC.md](SPEC.md) contains an earlier schedule with different dates and a
different order. The two have not been reconciled.
