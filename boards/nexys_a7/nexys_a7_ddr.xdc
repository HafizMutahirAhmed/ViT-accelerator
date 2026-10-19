## Constraints of boards/nexys_a7/nexys_top_ddr.sv on the Digilent Nexys A7-100T.
##
## Every pin line below is a line of Digilent's Nexys-A7-100T-Master.xdc,
## uncommented, with only the port name in get_ports renamed to the top's port:
##   source  https://github.com/Digilent/digilent-xdc/blob/7583b4e7a6b8178afdb7f9e07f18162c48a89b0d/Nexys-A7-100T-Master.xdc
##   commit  7583b4e7a6b8178afdb7f9e07f18162c48a89b0d (2024-04-15), the last
##           commit to touch the file; repository master c8644a9fe31c
##   sha256  40b8d2b557673d90fbb20018210e4e3460c52e071696edaebd4b6ad5b00599e2
## Renamed: CLK100MHZ (kept), CPU_RESETN -> ck_rst, UART_TXD_IN -> uart_txd_in,
##          UART_RXD_OUT -> uart_rxd_out, LED[0..4] -> led[0..4].
## UART names are from the USB-UART bridge's point of view: the FPGA receives
## on uart_txd_in (C4) and transmits on uart_rxd_out (D4).
## The master's create_clock line for CLK100MHZ is left out, because the clocking
## wizard (system_clock_generator) constrains its input clock itself.
## The DDR2 pins are not here: the MIG sets them from boards/nexys_a7/mig_ddr2.prj.

## Clock signal
set_property -dict { PACKAGE_PIN E3    IOSTANDARD LVCMOS33 } [get_ports { CLK100MHZ }]; #IO_L12P_T1_MRCC_35 Sch=clk100mhz

## LEDs
set_property -dict { PACKAGE_PIN H17   IOSTANDARD LVCMOS33 } [get_ports { led[0] }]; #IO_L18P_T2_A24_15 Sch=led[0]
set_property -dict { PACKAGE_PIN K15   IOSTANDARD LVCMOS33 } [get_ports { led[1] }]; #IO_L24P_T3_RS1_15 Sch=led[1]
set_property -dict { PACKAGE_PIN J13   IOSTANDARD LVCMOS33 } [get_ports { led[2] }]; #IO_L17N_T2_A25_15 Sch=led[2]
set_property -dict { PACKAGE_PIN N14   IOSTANDARD LVCMOS33 } [get_ports { led[3] }]; #IO_L8P_T1_D11_14 Sch=led[3]
set_property -dict { PACKAGE_PIN R18   IOSTANDARD LVCMOS33 } [get_ports { led[4] }]; #IO_L7P_T1_D09_14 Sch=led[4]

##CPU Reset Button
set_property -dict { PACKAGE_PIN C12   IOSTANDARD LVCMOS33 } [get_ports { ck_rst }]; #IO_L3P_T0_DQS_AD1P_15 Sch=cpu_resetn

##USB-RS232 Interface
set_property -dict { PACKAGE_PIN C4    IOSTANDARD LVCMOS33 } [get_ports { uart_txd_in }]; #IO_L7P_T1_AD6P_35 Sch=uart_txd_in
set_property -dict { PACKAGE_PIN D4    IOSTANDARD LVCMOS33 } [get_ports { uart_rxd_out }]; #IO_L11N_T1_SRCC_35 Sch=uart_rxd_out
