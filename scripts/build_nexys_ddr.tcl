# Synthesis, implementation and bitstream of build/nexys_a7_ddr_project in batch,
# with the timing and utilization reports in build/nexys_a7_ddr_reports.
#
#   vivado -mode batch -source scripts/create_nexys_ddr_project.tcl
#   vivado -mode batch -source scripts/build_nexys_ddr.tcl
#
# JOBS parallel runs (default 4). The bitstream is copied to build/nexys_top_ddr.bit,
# the file to program the board with; the project can be deleted after that.

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file dirname $script_dir]
set project_dir [file join $repo_dir build nexys_a7_ddr_project]
set report_dir  [file join $repo_dir build nexys_a7_ddr_reports]
set jobs [expr {[info exists ::env(JOBS)] ? $::env(JOBS) : 4}]

if {[llength [get_projects -quiet]] > 0} {
    close_project
}
open_project [file join $project_dir nexys_a7_ddr.xpr]
reset_run synth_1
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    error "build_nexys_ddr: synthesis failed, see [get_property DIRECTORY [get_runs synth_1]]"
}
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "build_nexys_ddr: implementation failed, see [get_property DIRECTORY [get_runs impl_1]]"
}
open_run impl_1
file mkdir $report_dir
report_timing_summary -max_paths 10 -file [file join $report_dir timing_summary.txt]
report_utilization -file [file join $report_dir utilization.txt]
report_clock_interaction -file [file join $report_dir clock_interaction.txt]
report_cdc -file [file join $report_dir cdc.txt]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "build_nexys_ddr: WNS $wns ns, WHS $whs ns, reports in $report_dir"
file copy -force [file join [get_property DIRECTORY [get_runs impl_1]] nexys_top_ddr.bit] \
    [file join $repo_dir build nexys_top_ddr.bit]
puts "build_nexys_ddr: bitstream [file join $repo_dir build nexys_top_ddr.bit]"
