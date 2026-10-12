# Schedule

Two fixed dates:

| Date | What |
|---|---|
| Oct 20, 2026 | Hardware bring-up of the matmul engine must have started by this day |
| Nov 2, 2026 | Project deadline |

## Where the project is

The matmul engine is frozen at the tag `engine-v1.1` and described in
[ENGINE.md](ENGINE.md). It is bit-exact in simulation on all 39 matmul ops of
the two DeiT-Tiny layers, the patch embedding and the classifier. Nothing has
run on the board yet.

## Order of work

1. **Non-linear ops.** Softmax, LayerNorm, GELU and the residual add, each with
   its own golden vectors and self-checking testbench.
2. **Full block.** One encoder layer end to end: the engine ops and the
   non-linear ops in sequence, bit-exact against the golden model.
3. **Loop x2.** The same block run twice, for layers 0 and 1, with the weights
   of each layer.
4. **Classifier.** The linear head on the CLS row and the argmax.
5. **Integration.** Everything together on the board.

Hardware bring-up of the engine is not a step in this list. It runs alongside
it and has to start by Oct 20, whichever step is in progress then.

[SPEC.md](SPEC.md) contains an earlier schedule with different dates and a
different order. The two have not been reconciled.
