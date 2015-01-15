__author__ = "Bar Bokovza"

import Code.dataHolder as dat
import Code.compiler as comp

data = dat.GAP_Data()
data.Load("External/Data/fb-net1.csv")
data.Load("External/Data/fb-net2.csv")
data.Load("External/Data/fb-net3.csv")

com = comp.GAP_Compiler()
com.Load("External/Rules/Pi4i.gap")
test = 1

