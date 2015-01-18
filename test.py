__author__ = "Bar Bokovza"

import numpy as np

import Code.dataHolder as dat
import Code.compiler as comp
import Code.python_relations as relation

#import time

"""
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
        elif a * b * c * d >= MainDict["g1_member"][(a_0,)] + 1e-05:
            changed += 1
            MainDict["g1_member"][(a_0,)] = a * b * c * d
    return added, changed

def Rule_1 (def_zone:tuple) -> tuple:
    assigns, varsPic = def_zone
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        a_1 = row[varsPic[1]]
        b = MainDict["g1_member"][(a_1,)]
        c = MainDict["p"][(a_1,)]
        d = MainDict["q"][(a_0,)]
        a = MainDict["friend"][(a_1, a_0,)]
        if (a_0,) not in MainDict["g1_member"].keys() and a * b * c * d > 0:
            added += 1
            MainDict["g1_member"][(a_0,)] = a * b * c * d
        elif a * b * c * d >= MainDict["g1_member"][(a_0,)] + (1e-05):
            changed += 1
            MainDict["g1_member"][(a_0,)] = a * b * c * d
    return added, changed

def Rule_2 (def_zone:tuple) -> tuple:
    assigns, varsPic = def_zone
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        a_1 = row[varsPic[1]]
        b = MainDict["g2_member"][(a_1,)]
        c = MainDict["q"][(a_1,)]
        d = MainDict["p"][(a_0,)]
        a = MainDict["friend"][(a_1, a_0,)]
        if (a_0,) not in MainDict["g2_member"].keys() and a * b * c * d > 0:
            added += 1
            MainDict["g2_member"][(a_0,)] = a * b * c * d
        elif a * b * c * d >= MainDict["g2_member"][(a_0,)] + 1e-05:
            changed += 1
            MainDict["g2_member"][(a_0,)] = a * b * c * d
    return added, changed

def Rule_3 (def_zone:tuple) -> tuple:
    assigns, varsPic = def_zone
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        a_1 = row[varsPic[1]]
        b = MainDict["g2_member"][(a_1,)]
        c = MainDict["q"][(a_1,)]
        d = MainDict["q"][(a_0,)]
        a = MainDict["friend"][(a_1, a_0,)]
        if (a_0,) not in MainDict["g2_member"].keys() and a * b * c * d > 0:
            added += 1
            MainDict["g2_member"][(a_0,)] = a * b * c * d
        elif a * b * c * d >= MainDict["g2_member"][(a_0,)] + 1e-05:
            changed += 1
            MainDict["g2_member"][(a_0,)] = a * b * c * d
    return added, changed

def Rule_4 (def_zone:tuple) -> tuple:
    assigns, varsPic = def_zone
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        b = MainDict["g2_member"][(a_0,)]
        c = MainDict["p"][(a_0,)]
        d = MainDict["q"][(a_0,)]
        if (a_0,) not in MainDict["g1_member"].keys() and b * c * d > 0:
            added += 1
            MainDict["g1_member"][(a_0,)] = b * c * d
        elif b * c * d >= MainDict["g1_member"][(a_0,)] + 1e-05:
            changed += 1
            MainDict["g1_member"][(a_0,)] = b * c * d
    return added, changed

def Rule_5 (def_zone:tuple) -> tuple:
    assigns, varsPic = def_zone
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        b = MainDict["g1_member"][(a_0,)]
        c = MainDict["q"][(a_0,)]
        d = MainDict["p"][(a_0,)]
        if (a_0,) not in MainDict["g2_member"].keys() and b * c * d > 0:
            added += 1
            MainDict["g2_member"][(a_0,)] = b * c * d
        elif b * c * d >= MainDict["g2_member"][(a_0,)] + 1e-05:
            changed += 1
            MainDict["g2_member"][(a_0,)] = b * c * d
    return added, changed
"""

def Check_Rule (rule:comp.GAP_Rule, def_zone:tuple) -> list:
    assigns, varsPic = def_zone

    count = []
    virtual = []
    for assign in assigns:
        virtual.clear()
        for i in range(np.shape(varsPic)[0]):
            virtual.append(assign[varsPic[i]])

        for block in rule.Body:
            tup = tuple()
            for idx in block.VirtualVarsPic:
                tup += (virtual[idx],)

            if tup not in MainDict[block.Predicat].keys():
                count.append(tup)

    return count






dataHold = dat.GAP_Data()
dataHold.Load("External/Data/fb-net1.csv")
dataHold.Load("External/Data/fb-net2.csv")
dataHold.Load("External/Data/fb-net3.csv")

com = comp.GAP_Compiler()
com.Load("External/Rules/Pi4a.gap")

gpu = relation.GAP_PythonRelations()

MainDict = dataHold.data

def_zones = []
idx = 5
rule = com.Rules[5]

def_zones.append(rule.Create_DefinitionZone(dataHold, gpu))
rule.Arrange_Execution(idx)
print("Rule #{0} => {1}".format(idx, Check_Rule(rule, def_zones[-1])))
idx += 1


#add0, change0 = Rule_0(def_zones[0])
#add1, change1 = Rule_1(def_zones[1])
#res = Rule_2_Check(def_zones[2])
#add2, change2 = Rule_2(def_zones[2])
#add3, change3 = Rule_3(def_zones[3])
#add4, change4 = Rule_4(def_zones[4])
#add5, change5 = Rule_5(def_zones[5])

exit()

