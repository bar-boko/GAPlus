__author__ = "Bar Bokovza"

import Code.dataHolder as dat
import Code.compiler as comp
import Code.opencl as para

dataHold = dat.GAP_Data()
dataHold.Load("External/Data/fb-net1.csv")
dataHold.Load("External/Data/fb-net2.csv")
dataHold.Load("External/Data/fb-net3.csv")

com = comp.GAP_Compiler()
com.Load("External/Rules/Pi4i.gap")

gpu = para.GAP_OpenCL()

MainDict = dataHold.data

def_zone = []
for rule in com.Rules:
    def_zone.append(rule.Create_DefinitionZone(dataHold, gpu))

exit()
