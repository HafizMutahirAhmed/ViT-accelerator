# Creates the block design memory_system in the open project: the external
# memory of boards/nexys_a7/nexys_top_ddr.sv. Sourced by
# scripts/create_nexys_ddr_project.tcl, which sets repo_dir first.
#
#   mig            Memory Interface Generator, the Nexys A7's DDR2, configured by
#                  boards/nexys_a7/mig_ddr2.prj; AXI4 slave, 128 bits, ui_clk
#   dma            AXI DMA, simple mode (no scatter-gather), MM2S only: DDR to a
#                  64-bit stream; registers on AXI4-Lite
#   interconnect   SmartConnect: S00 the host (S_AXI_HOST), S01 the DMA's reads;
#                  M00 the DMA's registers, M01 the MIG. It crosses from clk_100
#                  to the MIG's ui_clk.
#
# Address map, the same for the host and the DMA:
#   0x4040_0000  64 KiB   DMA registers (host only)
#   0x8000_0000  128 MiB  DDR2
#
# External ports (the wrapper memory_system_wrapper has them by these names):
#   clk_100, system_aresetn  clock and active-low reset of S_AXI_HOST, M_AXIS_DMA,
#                            the DMA and the interconnect; also the MIG's system clock
#   clk_200                  the MIG's 200 MHz reference clock
#   memory_resetn            the MIG's sys_rst, active low
#   memory_aresetn           the MIG's AXI reset, active low, synchronous to ui_clk
#   ui_clk, ui_clk_sync_rst  the MIG's user clock and its reset
#   init_calib_complete      the MIG has calibrated the DDR2
#   S_AXI_HOST               AXI4-Lite slave, 32-bit address and data
#   M_AXIS_DMA               the DMA's MM2S stream, 64 bits
#   DDR2                     the memory pins

set bd_name memory_system
create_bd_design $bd_name

# ---------------------------------------------------------------- MIG
set mig [create_bd_cell -type ip -vlnv xilinx.com:ip:mig_7series:4.2 mig]
set mig_component [get_property CONFIG.Component_Name $mig]
set mig_dir [get_property IP_DIR [get_ips $mig_component]]
set prj_in [open [file join $repo_dir boards nexys_a7 mig_ddr2.prj] r]
set prj_text [read $prj_in]
close $prj_in
regsub {<ModuleName>[^<]*</ModuleName>} $prj_text "<ModuleName>$mig_component</ModuleName>" prj_text
file mkdir $mig_dir
set prj_out [open [file join $mig_dir mig_ddr2.prj] w]
puts -nonewline $prj_out $prj_text
close $prj_out
set_property -dict [list CONFIG.XML_INPUT_FILE {mig_ddr2.prj}] $mig

# ---------------------------------------------------------------- DMA
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma]
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
] $dma

# ---------------------------------------------------------------- interconnect
set interconnect [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 interconnect]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {2} CONFIG.NUM_CLKS {2}] $interconnect

# ---------------------------------------------------------------- ports
set clk_100 [create_bd_port -dir I -type clk -freq_hz 100000000 clk_100]
set clk_200 [create_bd_port -dir I -type clk -freq_hz 200000000 clk_200]
set system_aresetn [create_bd_port -dir I -type rst system_aresetn]
set_property CONFIG.POLARITY ACTIVE_LOW $system_aresetn
set memory_resetn [create_bd_port -dir I -type rst memory_resetn]
set_property CONFIG.POLARITY ACTIVE_LOW $memory_resetn
set memory_aresetn [create_bd_port -dir I -type rst memory_aresetn]
set_property CONFIG.POLARITY ACTIVE_LOW $memory_aresetn
# memory_aresetn is made synchronous to ui_clk in nexys_top_ddr
set ui_clk [create_bd_port -dir O -type clk ui_clk]
set_property CONFIG.ASSOCIATED_RESET {memory_aresetn} $ui_clk
create_bd_port -dir O -type rst ui_clk_sync_rst
create_bd_port -dir O init_calib_complete

set host [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI_HOST]
set_property -dict [list \
    CONFIG.PROTOCOL {AXI4LITE} \
    CONFIG.ADDR_WIDTH {32} \
    CONFIG.DATA_WIDTH {32} \
    CONFIG.FREQ_HZ {100000000} \
] $host
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 M_AXIS_DMA
set_property -dict [list \
    CONFIG.ASSOCIATED_BUSIF {S_AXI_HOST:M_AXIS_DMA} \
    CONFIG.ASSOCIATED_RESET {system_aresetn} \
] $clk_100

# ---------------------------------------------------------------- connections
connect_bd_intf_net [get_bd_intf_ports S_AXI_HOST]   [get_bd_intf_pins interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_MM2S] [get_bd_intf_pins interconnect/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins interconnect/M00_AXI] [get_bd_intf_pins dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins interconnect/M01_AXI] [get_bd_intf_pins mig/S_AXI]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S] [get_bd_intf_ports M_AXIS_DMA]
make_bd_intf_pins_external [get_bd_intf_pins mig/DDR2]
set_property name DDR2 [get_bd_intf_ports -of_objects [get_bd_intf_nets -of_objects [get_bd_intf_pins mig/DDR2]]]

connect_bd_net [get_bd_ports clk_100] [get_bd_pins interconnect/aclk] [get_bd_pins dma/s_axi_lite_aclk] \
    [get_bd_pins dma/m_axi_mm2s_aclk] [get_bd_pins mig/sys_clk_i]
connect_bd_net [get_bd_ports clk_200] [get_bd_pins mig/clk_ref_i]
connect_bd_net [get_bd_pins mig/ui_clk] [get_bd_pins interconnect/aclk1] [get_bd_ports ui_clk]
connect_bd_net [get_bd_ports system_aresetn] [get_bd_pins interconnect/aresetn] [get_bd_pins dma/axi_resetn]
connect_bd_net [get_bd_ports memory_resetn] [get_bd_pins mig/sys_rst]
connect_bd_net [get_bd_ports memory_aresetn] [get_bd_pins mig/aresetn]
connect_bd_net [get_bd_pins mig/ui_clk_sync_rst] [get_bd_ports ui_clk_sync_rst]
connect_bd_net [get_bd_pins mig/init_calib_complete] [get_bd_ports init_calib_complete]

# ---------------------------------------------------------------- addresses
assign_bd_address -offset 0x40400000 -range 64K -target_address_space [get_bd_addr_spaces S_AXI_HOST] \
    [get_bd_addr_segs dma/S_AXI_LITE/Reg] -force
assign_bd_address -offset 0x80000000 -range 128M -target_address_space [get_bd_addr_spaces S_AXI_HOST] \
    [get_bd_addr_segs mig/memmap/memaddr] -force
assign_bd_address -offset 0x80000000 -range 128M -target_address_space [get_bd_addr_spaces dma/Data_MM2S] \
    [get_bd_addr_segs mig/memmap/memaddr] -force
# the DMA reads DDR only, never its own registers
exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces dma/Data_MM2S] [get_bd_addr_segs dma/S_AXI_LITE/Reg]

validate_bd_design
save_bd_design
