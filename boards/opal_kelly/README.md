# Opal Kelly XEM7310-A200

Not built yet. This directory will hold what is specific to the board, and
nothing else: the top level, the FrontPanel adapter that turns pipes into the
byte stream of [docs/HOST_PROTOCOL.md](../../docs/HOST_PROTOCOL.md), the
constraints and the build script. The accelerator behind it is the one in
`rtl/core` and `rtl/interface`, unchanged.

Opal Kelly's FrontPanel HDL (`okHost`, `okLibrary`, the pipes) and its
simulation models are proprietary. They are not kept in this repository;
the build will take them from a FrontPanel SDK installed outside it.
