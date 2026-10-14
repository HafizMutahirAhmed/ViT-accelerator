## Configuration bank voltage of the Nexys A7-100T.
##
## These two properties are NOT from Digilent's Nexys-A7-100T-Master.xdc, which
## does not set them; they are here in a file of their own for that reason.
## They tell Vivado that the configuration bank (bank 0) is powered at 3.3 V
## and they remove the DRC warning CFGBVS-1. They change no pin.
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
