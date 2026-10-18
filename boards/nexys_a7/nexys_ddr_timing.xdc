## Timing constraints of boards/nexys_a7/nexys_top_ddr.sv that are not pins.
##
## The MIG's init_calib_complete (ui_clk domain) enters the CLK domain through
## the two-flip-flop synchroniser calibrated_sync (ASYNC_REG); it is a level
## that changes once after reset, so the path into the first flip-flop is not
## timed. The interconnect's own clock crossing carries its own constraints.
set_false_path -to [get_pins -hierarchical -filter {NAME =~ *calibrated_sync_reg[0]/D}]
