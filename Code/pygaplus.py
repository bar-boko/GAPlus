__author__ = "Bar Bokovza"

#region Imports
import numpy as np

import Code.compiler as com
import Code.dataHolder as holder

#from Code.basic import GAP_Basic
from Code.opencl import GAP_OpenCL
import sys
#import time
#import gc
#endregion

#region Data
dataHold = holder.GAP_Data()
comp = com.GAP_Compiler()

#prev, next = { }, { }
intervals = 0
MainDict = { }

gpu = GAP_OpenCL("../External/OpenCL/Commands.cl")
#gpu = GAP_Basic()

#addedLst, changedLst = [], [] ## for predicats
#toDefZone, toRun = [], []     ## for rules
fix_point = False
add_fix_point = False
changeSet = []

def_zones = []
#endregion

#region Help Commands
print("GAPlus - GAP Compiler using OpenCL - By Bar Bokovza")
print("===================================================")
print("Commands :")
print("===================================================")
print("Load_Data(path:string)   - Load data to the Data Holder")
print("Load_Rules(path:string)  - Load the rules from file")
print("---------------------------------------------------")
print("Add(rule:str)            - Add a rule to the compiler")
print("---------------------------------------------------")
print("Run()                    - Execute 1 times the GAP Rules")
print("Run_FixPoint()           - Run until fix")
print("---------------------------------------------------")
print("Export_Data(path:str)    - Export the data from the engine to a csv file")
print("Export_Rules([path:str]) - Export the compiled code from the engine to a file")
print("Export_Rules()           - Prints the compiled")
print("---------------------------------------------------")
print("Reset()                  - Reset all the console")
print("Reset_Data()             - Reset only the data from the console")
print("Reset_Rules()            - Reset only the GAP rules from the console")
print("---------------------------------------------------")
print("Exit()                   - Close the prompt.")
print("---------------------------------------------------")
print("<command>                - Any accepted python 2.7.x command.")
print("===================================================")
#endregion

#region Functions
def Load_Data (path):
    """
    Loaded the csv file in the path to the data holder.
    :type path: str
    :param path: csv file path that contains the data
    :rtype: void
    """
    dataHold.Load(path)

def Load_Rules (path):
    """
    Load the rules in the GAP file to the console.
    :type path: str
    :param path: GAP file
    :rtype: void
    """
    comp.Load(path)

"""
def PreRun ():
    global lstRules, lstRules_Ground
    global MainDict

    MainDict = dataHold.data
    for predicat in comp.GetPredicats():
        next[predicat] = 1, 1
    for i in range(len(comp.Rules)):
        rule = comp.Rules[i]
        changeSet.append((0, 0))
        rule.Arrange_Execution(i, 0)
        def_zones.append((np.zeros(0, dtype = np.int32), np.zeros(0, dtype = np.int32)))
        exec(rule.Code_Run, globals())
"""

def PreRun ():
    """
    Preparing the console before running the code in the first time.
    """
    global MainDict

    MainDict = dataHold.data
    for i in range(len(comp.Rules)):
        rule = comp.Rules[i]
        changeSet.append((0, 0))
        rule.Arrange_Execution(i, 0)
        def_zones.append((np.zeros(0, dtype = np.int32), np.zeros(0, dtype = np.int32)))
        exec(rule.Code_Run, globals())

def Interval ():
    """
    Execute all the rules in the compiler once.
    :return: void
    """
    global comp
    global dataHolder
    global fix_point, add_fix_point
    global intervals

    #toDefZone.clear(), toRun.clear(), addedLst.clear(), changedLst.clear()

    added, changed = 0, 0

    for i in range(len(comp.Rules)):
        rule = comp.Rules[i]
        if not add_fix_point:
            def_zones[i] = rule.Create_DefinitionZone(dataHold, gpu)
        exec(compile("Rule_{0}(def_zones[{0}], changeSet, {0})".format(i), "<string>", "exec"))
        add, change = changeSet[i]
        #print("#{0} -> {1},{2}".format(i, added, changed))
        added += add
        changed += changed

    if added == 0:
        if changed == 0:
            fix_point = True
            return
        else:
            add_fix_point = True

    intervals += 1

"""
def Interval ():
    global comp
    global dataHolder
    global fix_point
    global intervals

    toDefZone.clear(), toRun.clear(), addedLst.clear(), changedLst.clear()

    rules = lstRules
    if intervals == 0:
        rules += lstRules_Ground

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

            exec(compile("Rule_{0}(def_zones[{0}], changeSet, i)".format(i), "<string>", "exec"))
            next_add, next_change = next[comp.Rules[i].Header.Predicat]
            added, changed = changeSet[i]
            #print("Rule #{0} => Added : {1}, Changed : {2}".format(i, added, changed))
            next[comp.Rules[i].Header.Predicat] = (next_add + added, next_change + changed)

    intervals += 1
"""

def Run ():
    """
    Execute Single Interval
    """
    if intervals is 0:
        PreRun()

    Interval()

def Run_FixPoint ():
    """
    Run the GAP rules until reaching a fix point
    """
    if intervals is 0:
        PreRun()

    while not fix_point:
        Interval()

def Reset ():
    """
    Clean all the console from all the rules and data.
    """
    global MainDict, fix_point, changeSet, def_zones, intervals, add_fix_point
    #global prev, next, addedLst, changedLst, toDefZone, toRun

    Reset_Data()
    Reset_Rules()

    #prev.clear()
    #next.clear()
    #addedLst.clear()
    #changedLst.clear()
    #toDefZone.clear()
    #toRun.clear()

    fix_point = False
    add_fix_point = False

def Reset_Data ():
    """
    Clear the data holder.
    """
    global intervals, def_zones, changeSet
    dataHold.Reset()
    MainDict.clear()
    def_zones.clear()
    changeSet.clear()
    intervals = 0

def Reset_Rules ():
    """
    Clear the compiler from GAP Rules
    """
    global intervals
    comp.Reset()
    intervals = 0

def Exit ():
    """
    Exit the console.
    """
    sys.exit()

"""
def Benchmark (dataPath:str = "../External/Data/fb-net1.csv", rulesPath:str = "../External/Rules/Pi4a.gap",
        max:int = 11):
    result = []
    count = 1

    print("#> STARTED ! GOOD LUCK !")

    while count <= max:
        Reset()
        t_start = time.time()
        dataHold.Load(dataPath)
        t_end = time.time()
        t_data = t_end - t_start

        t_start = time.time()
        comp.Load(rulesPath)
        t_end = time.time()
        t_compile = t_end - t_start

        t_start = time.time()
        Basic_Run_FixPoint()
        t_end = time.time()
        t_run = t_end - t_start

        print("{0},{1},{2},{3}".format(t_data, t_compile, t_run, t_data + t_compile + t_run))

        result.append((t_data, t_compile, t_run))

        count += 1

        if count == 1:
            gc.collect()

    print("#> ENDED ! RESULTS ARE IN THE LOG.CSV FILE")
    Exit()
"""

#Benchmark(max = 100)

while True:
    command = sys.stdin.readline()
    try:
        exec(command)
    except Exception as e:
        print("> Exception : {0}".format(e))
    finally:
        print("> Done !")
