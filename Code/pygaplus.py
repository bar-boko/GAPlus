import sys
import time

__author__ = "Bar Bokovza"

#region Imports
import numpy as np
import Code.compiler as com
import Code.dataHolder as holder
import Code.opencl as cl
#endregion

dataHold = holder.GAP_Data()
comp = com.GAP_Compiler()

prev, next = { }, { }
intervals = 0
MainDict = { }

gpu = cl.GAP_OpenCL("../External/OpenCL/Commands.cl")

addedLst, changedLst = [], [] ## for predicats
toDefZone, toRun = [], []     ## for rules
fix_point = False

def_zones = []

print("GAPlus - GAP Compiler using OpenCL - By Bar Bokovza")
print("===================================================")
print("Commands :")
print("---------------------------------------------------")
print("Load_Data(path:string) - Load data to the Data Holder")
print("Load_Rules(path:string) - Load the rules from file")
print("Add(rule:str) - Add a rule to the compiler")
print("Run() - Execute 1 times the GAP Rules.")
print("Run_FixPoint() - Run until fix")
print("---------------------------------------------------")
print("Exit() - Close the prompt.")
print("<command> - Any accepted python 3.4.x command.")
print("---------------------------------------------------")

#region Functions
def Load_Data (path:str):
    dataHold.Load(path)

def Load_Rules (path:str):
    comp.Load(path)

def PreRun ():
    global MainDict

    MainDict = dataHold.data
    for predicat in comp.GetPredicats():
        next[predicat] = 1, 1
    for i in range(len(comp.Rules)):
        comp.Rules[i].Arrange_Execution(i, 0)
        def_zones.append((np.zeros(0, dtype = np.int32), np.zeros(0, dtype = np.int32)))

def Interval ():
    global comp
    global dataHolder
    global fix_point

    toDefZone.clear(), toRun.clear(), addedLst.clear(), changedLst.clear()

    for predicat in next.keys():
        prev[predicat] = next[predicat]
        added, changed = prev[predicat]
        next[predicat] = 0, 0

        if added > 0:
            addedLst.append(predicat)
        elif changed > 0:
            changedLst.append(predicat)

    # arranging predicats
    for i in range(len(comp.Rules)):
        rule = comp.Rules[i]
        for predicat in addedLst:
            if predicat in rule.Predicats_Dependent:
                toDefZone.append(i)
                break

        for predicat in changedLst:
            if predicat in rule.Predicats_Dependent:
                toRun.append(i)
                break

    if len(toDefZone) + len(toRun) == 0:
        fix_point = True
        return

    # Starting execution here
    if len(toDefZone) > 0:
        for i in toDefZone:
            def_zones[i] = comp.Rules[i].Create_DefinitionZone(dataHold, gpu)
            if not com._IsEmpty(def_zones[i][0]):
                toRun.append(i)

    for i in toRun:
        if np.shape((def_zones[i])[0])[0] > 0:
            added, changed = 0, 0
            exec(comp.Rules[i].Code_Run)
            exec(compile("added, changed = Rule_{0}(def_zones[{0}])".format(i), "<string>", "exec"))

            next_add, next_change = next[comp.Rules[i].Header.Predicat]
            next[comp.Rules[i].Header.Predicat] = (next_add + added, next_change + changed)

def Run ():
    if intervals is 0:
        PreRun()

    Interval()

def Run_FixPoint ():
    if intervals is 0:
        PreRun()

    while not fix_point:
        Interval()

def Exit ():
    sys.exit()

time_begin = time.time()
Load_Data("../External/Data/fb-net1.csv")
time_end = time.time()
print("Data : {0}s".format(time_end - time_begin))

time_begin = time.time()
Load_Rules("../External/Rules/Pi4a.gap")
time_end = time.time()
print("Compile : {0}s".format(time_end - time_begin))

time_begin = time.time()
Run_FixPoint()
time_end = time.time()
print("Runtime : {0}s".format(time_end - time_begin))


