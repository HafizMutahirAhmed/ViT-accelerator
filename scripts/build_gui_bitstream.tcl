# Does what Generate Bitstream does in vivado/vit.xpr, then writes to <out_dir>: nexys_top_ddr.bit,
# build.txt ("WNS <ns> WHS <ns>"), timing_summary.txt and utilization.txt.
#
# Batch: vivado -mode batch -source scripts/build_gui_bitstream.tcl -tclargs <project.xpr> <out_dir>

lassign $argv project_file out_dir
set jobs [expr {[info exists ::env(JOBS)] ? $::env(JOBS) : 4}]
file mkdir $out_dir
open_project $project_file
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "build_gui_bitstream: implementation failed, see [get_property DIRECTORY [get_runs impl_1]]"
}
open_run impl_1
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
set out [open [file join $out_dir build.txt] w]
puts $out "WNS $wns WHS $whs"
close $out
report_timing_summary -max_paths 10 -file [file join $out_dir timing_summary.txt]
report_utilization -file [file join $out_dir utilization.txt]
file copy -force [file join [get_property DIRECTORY [get_runs impl_1]] nexys_top_ddr.bit] $out_dir
puts "build_gui_bitstream: WNS $wns ns, WHS $whs ns, [file join $out_dir nexys_top_ddr.bit]"
close_project
