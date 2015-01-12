__author__ = "Bar Bokovza"

import Code.dataHolder as dat
import Code.compiler as comp

data = dat.GAP_Data( )
data.load( "External/Data/fb-net1.csv" )
data.load( "External/Data/fb-net2.csv" )
data.load( "External/Data/fb-net3.csv" )

compile = comp.GAP_Compiler( )
compile.load( "External/Rules/Pi4i.gap" )
test = 1

