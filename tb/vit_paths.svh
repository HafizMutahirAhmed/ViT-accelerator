//////////////////////////////////////////////////////////////////////////////////
// Where the testbenches find the repository: VIT_ROOT is the repository's top directory,
// relative to the directory the simulator runs in.
//   the Vivado project vivado/vit.xpr runs in vivado/vit.sim/<set>/behav/xsim/ and every
//   simulation set defines VIT_VIVADO_PROJECT                               VIT_ROOT "../../../../../"
//   a simulation run from the command line in sim/<name>/                   VIT_ROOT "../../"
// The vectors are under VIT_ROOT "python/", the ROM images under VIT_ROOT "rtl/rom/input_<size>/".
//////////////////////////////////////////////////////////////////////////////////

`ifndef VIT_PATHS_SVH
`define VIT_PATHS_SVH

`ifdef VIT_VIVADO_PROJECT
`define VIT_ROOT "../../../../../"
`else
`define VIT_ROOT "../../"
`endif

// A ROM image of the 224 x 224 or the 112 x 112 configuration, e.g. `VIT_ROM_224("exp_rom.hex").
`define VIT_ROM_224(name) {`VIT_ROOT, "rtl/rom/input_224/", name}
`define VIT_ROM_112(name) {`VIT_ROOT, "rtl/rom/input_112/", name}

`endif
