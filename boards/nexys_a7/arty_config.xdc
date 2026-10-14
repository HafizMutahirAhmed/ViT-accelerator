## Configuration bank voltage of the Arty A7.
##
## These two properties are NOT from Digilent's Arty-A7-100-Master.xdc, which
## does not set them; they are here in a file of their own for that reason.
## They tell Vivado that the configuration bank (bank 0) is powered at 3.3 V,
## which is what the Arty A7 does, and they remove the DRC warning CFGBVS-1.
## They change no pin.
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
