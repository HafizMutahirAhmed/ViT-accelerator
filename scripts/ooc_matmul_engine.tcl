# Out-of-context synthesis and implementation of matmul_engine.
#
# Usage: vivado -mode batch -source scripts/ooc_matmul_engine.tcl
#        vivado -mode batch -source scripts/ooc_matmul_engine.tcl -tclargs <clock period ns>
#
# The part comes from ViT-implementation.xpr. The engine is built with its
# package defaults. Reports and the routed checkpoint go to
# sim/ooc_matmul_engine/ (gitignored):
#   timing_summary.rpt, timing_worst10.rpt, worst_paths.txt
#   utilization.rpt, utilization_hier.rpt
#   dsp_synth.txt, dsp_route.txt   one line per DSP48E1: registers and OPMODE
#   memories.txt                   block RAM and LUT RAM cells by parent

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file dirname $script_dir]
set out_dir    [file join $repo_dir sim ooc_matmul_engine]
file mkdir $out_dir

set period 6.667
if {[llength $argv] > 0} {
    set period [lindex $argv 0]
}

set fh [open [file join $repo_dir ViT-implementation.xpr]]
regexp {Option Name="Part" Val="([^"]+)"} [read $fh] -> part
close $fh
puts "ooc_matmul_engine: part $part, clock period $period ns"

# one line per DSP48E1: register attributes and whether each OPMODE bit is a
# constant or a signal
proc report_dsps {file_name} {
    set fh [open $file_name w]
    puts $fh "cell AREG BREG CREG MREG PREG OPMODEREG USE_MULT OPMODE\[6:0\] (0/1 constant, d dynamic)"
    foreach cell [lsort -dictionary [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]] {
        set opmode ""
        for {set bit 6} {$bit >= 0} {incr bit -1} {
            set net [get_nets -quiet -of_objects [get_pins "$cell/OPMODE[$bit]"]]
            if {$net eq ""} {
                append opmode "x"
            } else {
                switch [get_property TYPE $net] {
                    GROUND  {append opmode "0"}
                    POWER   {append opmode "1"}
                    default {append opmode "d"}
                }
            }
        }
        puts $fh "$cell [get_property AREG $cell] [get_property BREG $cell] [get_property CREG $cell]\
[get_property MREG $cell] [get_property PREG $cell] [get_property OPMODEREG $cell]\
[get_property USE_MULT $cell] $opmode"
    }
    close $fh
}

# memory primitives counted by their parent instance
proc report_memories {file_name} {
    set fh [open $file_name w]
    array set count {}
    foreach cell [get_cells -hierarchical -filter {PRIMITIVE_GROUP == BLOCKRAM || PRIMITIVE_GROUP == BMEM || REF_NAME =~ RAMB* || REF_NAME =~ RAM*M* || REF_NAME =~ RAMD* || REF_NAME =~ RAMS*}] {
        set parent [get_property PARENT $cell]
        regsub -all {\[[0-9]+\]} $parent {[*]} parent
        set key "$parent [get_property REF_NAME $cell]"
        if {![info exists count($key)]} {set count($key) 0}
        incr count($key)
    }
    foreach key [lsort [array names count]] {
        puts $fh "$count($key) $key"
    }
    close $fh
}

set rtl_dir [file join $repo_dir rtl]
set package [file join $rtl_dir vit_config_pkg.sv]
set others  {}
foreach f [lsort [glob -directory $rtl_dir *.sv]] {
    if {$f ne $package} {lappend others $f}
}
read_verilog -sv [concat $package $others]

synth_design -top matmul_engine -part $part -mode out_of_context
create_clock -name CLK -period $period [get_ports CLK]
report_dsps [file join $out_dir dsp_synth.txt]
report_utilization -hierarchical -file [file join $out_dir utilization_hier_synth.rpt]

opt_design
place_design
phys_opt_design
route_design

report_timing_summary -file [file join $out_dir timing_summary.rpt]
report_timing -max_paths 10 -nworst 1 -sort_by slack -file [file join $out_dir timing_worst10.rpt]
report_utilization -file [file join $out_dir utilization.rpt]
report_utilization -hierarchical -file [file join $out_dir utilization_hier.rpt]
report_dsps [file join $out_dir dsp_route.txt]
report_memories [file join $out_dir memories.txt]

set fh [open [file join $out_dir worst_paths.txt] w]
puts $fh "slack_ns levels datapath_ns startpoint -> endpoint"
foreach path [get_timing_paths -max_paths 10 -nworst 1 -sort_by slack] {
    puts $fh "[get_property SLACK $path] [get_property LOGIC_LEVELS $path] [get_property DATAPATH_DELAY $path]\
[get_property STARTPOINT_PIN $path] -> [get_property ENDPOINT_PIN $path]"
}
close $fh

write_checkpoint -force [file join $out_dir matmul_engine_routed.dcp]
puts "ooc_matmul_engine: WNS [get_property SLACK [get_timing_paths -max_paths 1]] ns, reports in $out_dir"
