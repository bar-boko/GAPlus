__author__ = "Bar Bokovza"

import Code.dataHolder as dat

data = dat.GAP_Data( )
data.load( "External/Data/fb-net1.csv" )
data.load( "External/Data/fb-net2.csv" )
data.load( "External/Data/fb-net3.csv" )

item = data.to_NDArray( "friend" )
item2 = data.to_NDArray( "g1_member" )



