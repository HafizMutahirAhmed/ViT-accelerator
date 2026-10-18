# Generates the AXI DMA of the block design memory_system as a stand-alone IP,
# dma_ip, with the same configuration, and exports its xsim simulation sources
# to build/dma_ip_sim/export/xsim (vhdl.prj, vlog.prj). Used by
# scripts/sim_dma_ip.sh; keep the CONFIG list equal to
# scripts/create_memory_system_bd.tcl's.

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file dirname $script_dir]
set out_dir    [file join $repo_dir build dma_ip_sim]

file delete -force $out_dir
create_project -force dma_ip_sim [file join $out_dir project] -part xc7a100tcsg324-1
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
create_ip -name axi_dma -vendor xilinx.com -library ip -version 7.1 -module_name dma_ip
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_addr_width {32} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {64} \
    CONFIG.c_mm2s_burst_size {64} \
    CONFIG.c_include_mm2s_dre {0} \
] [get_ips dma_ip]
generate_target {simulation instantiation_template} [get_ips dma_ip]
export_ip_user_files -of_objects [get_ips dma_ip] -no_script -force \
    -ip_user_files_dir [file join $out_dir ip_user_files]
export_simulation -of_objects [get_ips dma_ip] -simulator xsim -force \
    -directory [file join $out_dir export] -ip_user_files_dir [file join $out_dir ip_user_files]
puts "export_dma_ip_sim: [file join $out_dir export xsim]"
