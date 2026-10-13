# Out-of-context synthesis and implementation of matmul_engine.
#
# Usage: vivado -mode batch -source scripts/ooc_matmul_engine.tcl
#        vivado -mode batch -source scripts/ooc_matmul_engine.tcl -tclargs <clock period ns> <array size>
#
# The part is the XEM7310-A200's, xc7a200tfbg484-1. The array is square, 8x8 unless
# an array size is given; everything else is the package default. The pool
# buffer depths are the real build's (buffer_depths below). Reports and the
# routed checkpoint go to sim/ooc_matmul_engine/, or
# sim/ooc_matmul_engine_<size>x<size>/ for another size (gitignored):
#   timing_summary.rpt, timing_worst10.rpt, worst_paths.txt
#   worst_by_kind.txt              worst path between each pair of modules
#   utilization.rpt, utilization_hier.rpt
#   dsp_synth.txt, dsp_route.txt   one line per DSP48E1: registers and OPMODE
#   memories.txt                   block RAM and LUT RAM cells by parent

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file dirname $script_dir]
set period 6.667
if {[llength $argv] > 0} {
    set period [lindex $argv 0]
}
set size 8
if {[llength $argv] > 1} {
    set size [lindex $argv 1]
}

set out_dir [file join $repo_dir sim ooc_matmul_engine]
if {$size != 8} {
    append out_dir "_${size}x${size}"
}
file mkdir $out_dir

set part xc7a200tfbg484-1
puts "ooc_matmul_engine: part $part, clock period $period ns, ${size}x${size} array"

# Words per bank of pool buffers 0..7: the plan of python/layer_program.py,
# for the same tensors at any array size. With T = ceil(197 / size) row tiles
# and C = ceil(64 / size) column tiles of one head:
#   0 X    residual stream               T * 192
#   1 W    one layer of weights          442,368 / size
#   2 BIG  197 x 768                     T * 768
#   3 N    197 x 192                     T * 192
#   4 S    scores of three heads         3 * T * 197
#   5 QV   Q and V of three heads        3 * T * 64 + 3 * C * 197
#   6 K    K of three heads              3 * T * 64
#   7      not used                      16
# At 8x8: 4800 55296 19200 4800 14775 9528 4800 16.
set token_tiles   [expr {(197 + $size - 1) / $size}]
set head_tiles    [expr {(64 + $size - 1) / $size}]
set buffer_depths [list \
    [expr {$token_tiles * 192}] \
    [expr {(442368 + $size - 1) / $size}] \
    [expr {$token_tiles * 768}] \
    [expr {$token_tiles * 192}] \
    [expr {3 * $token_tiles * 197}] \
    [expr {3 * $token_tiles * 64 + 3 * $head_tiles * 197}] \
    [expr {3 * $token_tiles * 64}] \
    16]
puts "ooc_matmul_engine: buffer depths $buffer_depths"
set depth_generic ""
foreach depth [lreverse $buffer_depths] {
    append depth_generic [format %08x $depth]
}
set depth_generic "[expr {32 * [llength $buffer_depths]}]'h$depth_generic"

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

# worst path between every pair of module groups, most negative first
proc report_worst_by_kind {file_name} {
    set groups {
        array      array/*
        requantize drain_inst/lane_requantize*
        drain      drain_inst/*
        controller controller/*
        pool       pool*
        bias_store bias_store/*
        top        *
    }
    # each cell goes to the first group whose pattern it matches
    array set members {}
    set taken {}
    foreach {name pattern} $groups {
        set cells {}
        foreach cell [get_cells -quiet -hierarchical -filter "NAME =~ $pattern && IS_PRIMITIVE && (IS_SEQUENTIAL || REF_NAME =~ RAMB* || REF_NAME == DSP48E1)"] {
            if {![dict exists $taken $cell]} {
                lappend cells $cell
                dict set taken $cell 1
            }
        }
        set members($name) $cells
    }
    set rows {}
    foreach {from _} $groups {
        foreach {to _} $groups {
            if {[llength $members($from)] == 0 || [llength $members($to)] == 0} continue
            set path [get_timing_paths -quiet -max_paths 1 -from $members($from) -to $members($to)]
            if {$path eq ""} continue
            lappend rows [list [get_property SLACK $path] "$from -> $to | slack [get_property SLACK $path]\
| levels [get_property LOGIC_LEVELS $path] | delay [get_property DATAPATH_DELAY $path]\
| [get_property STARTPOINT_PIN $path] -> [get_property ENDPOINT_PIN $path]"]
        }
    }
    set fh [open $file_name w]
    foreach row [lsort -real -index 0 $rows] {
        puts $fh [lindex $row 1]
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

set rtl_dir [file join $repo_dir rtl core]
set package [file join $rtl_dir vit_config_pkg.sv]
set others  {}
foreach f [lsort [glob -directory $rtl_dir *.sv]] {
    if {$f ne $package} {lappend others $f}
}
read_verilog -sv [concat $package $others]

synth_design -top matmul_engine -part $part -mode out_of_context \
    -generic "SYS_ROW=$size" -generic "SYS_COL=$size" -generic "BUFFER_DEPTH=$depth_generic"
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
report_worst_by_kind [file join $out_dir worst_by_kind.txt]

set fh [open [file join $out_dir worst_paths.txt] w]
puts $fh "slack_ns levels datapath_ns startpoint -> endpoint"
foreach path [get_timing_paths -max_paths 10 -nworst 1 -sort_by slack] {
    puts $fh "[get_property SLACK $path] [get_property LOGIC_LEVELS $path] [get_property DATAPATH_DELAY $path]\
[get_property STARTPOINT_PIN $path] -> [get_property ENDPOINT_PIN $path]"
}
close $fh

write_checkpoint -force [file join $out_dir matmul_engine_routed.dcp]
set wns [get_property SLACK [get_timing_paths -max_paths 1]]
puts "ooc_matmul_engine: WNS $wns ns, estimated Fmax [format %.1f [expr {1000.0 / ($period - $wns)}]] MHz, reports in $out_dir"
