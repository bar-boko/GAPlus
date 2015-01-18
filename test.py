__author__ = "Bar Bokovza"

import Code.dataHolder as dat
import Code.compiler as comp
import Code.opencl as para

def Rule_0 (def_zone:tuple) -> tuple:
    assigns, varsPic = def_zone
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        a_1 = row[varsPic[1]]
        b = MainDict["g1_member"][(a_1,)]
        c = MainDict["p"][(a_1,)]
        d = MainDict["p"][(a_0,)]
        a = MainDict["friend"][(a_1, a_0,)]
        if (a_0,) not in MainDict["g1_member"].keys() and a * b * c * d > 0:
            added += 1
            MainDict["g1_member"][(a_0,)] = a * b * c * d
        elif a * b * c * d >= MainDict["g1_member"][(a_0,)] + (1e-05):
            changed += 1
            MainDict["g1_member"][(a_0,)] = a * b * c * d
    return added, changed



dataHold = dat.GAP_Data()
dataHold.Load("External/Data/fb-net1.csv")
dataHold.Load("External/Data/fb-net2.csv")
dataHold.Load("External/Data/fb-net3.csv")

com = comp.GAP_Compiler()
com.Load("External/Rules/Pi4a.gap")

gpu = para.GAP_OpenCL()

MainDict = dataHold.data

def_zones = []
idx = 0
for rule in com.Rules:
    def_zones.append(rule.Create_DefinitionZone(dataHold, gpu))
    rule.Arrange_Execution(idx)
    idx += 1

exit()

