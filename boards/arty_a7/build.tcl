# Synthesis, implementation and bitstream of boards/arty_a7/arty_top.sv.
#
# Usage: vivado -mode batch -source boards/arty_a7/build.tcl [-tclargs <stop>]
#   <stop> is "route" to stop before the bitstream; the default writes it.
#
# Non-project flow for the Arty A7-100T (xc7a100tcsg324-1). Sources: every
# rtl/core/*.sv and rtl/interface/*.sv and the SystemVerilog files of this directory; constraints:
# arty_top.xdc. Everything goes to sim/arty_a7_build/ (gitignored):
#   arty_top.bit                 the bitstream
#   timing_summary.rpt, timing_worst10.rpt
#   utilization.rpt, utilization_hier.rpt
#   io.rpt                       the pins as placed
#   arty_top_routed.dcp

set board_dir [file dirname [file normalize [info script]]]
set repo_dir  [file dirname [file dirname $board_dir]]
set out_dir   [file join $repo_dir sim arty_a7_build]
file mkdir $out_dir
set stop bitstream
if {[llength $argv] > 0} {
    set stop [lindex $argv 0]
}

set rtl_dir [file join $repo_dir rtl core]
set package [file join $rtl_dir vit_config_pkg.sv]
set sources [list $package]
foreach f [lsort [glob -directory $rtl_dir *.sv]] {
    if {$f ne $package} { lappend sources $f }
}
foreach f [lsort [glob -directory [file join $repo_dir rtl interface] *.sv]] {
    lappend sources $f
}
foreach f [lsort [glob -directory $board_dir *.sv]] {
    lappend sources $f
}
read_verilog -sv $sources
read_xdc [file join $board_dir arty_top.xdc]

synth_design -top arty_top -part xc7a100tcsg324-1
report_utilization -file [file join $out_dir utilization_synth.rpt]
opt_design
place_design
phys_opt_design
route_design
phys_opt_design

report_timing_summary -file [file join $out_dir timing_summary.rpt]
report_timing -max_paths 10 -file [file join $out_dir timing_worst10.rpt]
report_utilization -file [file join $out_dir utilization.rpt]
report_utilization -hierarchical -file [file join $out_dir utilization_hier.rpt]
report_io -file [file join $out_dir io.rpt]
report_drc -file [file join $out_dir drc.rpt]
write_checkpoint -force [file join $out_dir arty_top_routed.dcp]

set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -hold]]
puts "arty_top: WNS $wns ns, WHS $whs ns"
if {$stop ne "route"} {
    if {$wns < 0 || $whs < 0} {
        puts "arty_top: timing is not met, no bitstream written"
    } else {
        write_bitstream -force [file join $out_dir arty_top.bit]
        puts "arty_top: wrote [file join $out_dir arty_top.bit]"
    }
}
