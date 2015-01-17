import sys

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
    dictList = ""
    MainDict = dataHold.data
    predicats = comp.GetPredicats()
    for predicat in predicats:
        next[predicat] = 1, 1
        dictList += "dict_{0} = MainDict[\"{0}\"]\n".format(predicat)
        dictList += "array_{0} = dataHold.Generate_NDArray(\"{0}\")\n".format(predicat)
    for i in range(len(comp.Rules)):
        comp.Rules[i].Arrange_Execution(i, 0)
        def_zones.append((None, None))

    exec(compile(dictList, "<string>", "exec"))

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

    if len(toDefZone) + len(toRun) is 0:
        fix_point = True
        return

    if len(toDefZone) > 0:
        for i in toDefZone:
            def_zone, varsPic = np.zeros(0, dtype = np.int32), np.zeros(0, dtype = np.int32)
            exec(comp.Rules[i].Code_DefinitionZone)
            exec(compile("def_zone, varsPic = DefinitionZone_{0}()".format(i), "<string>", "exec"))
            def_zones[i] = (def_zone, varsPic)

            toRun.append(i)

    for i in toRun:
        added, changed = 0, 0
        exec(comp.Rules.Code_Run)
        exec(compile("added, changed = Rule_{0}()".format(i), "<string>", "exec"))

        next_add, next_change = next[comp.Rules[i].Header.Predicat]
        next[comp.Rules[i].Header.Predicat] = next_add + added, next_change + changed

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

Load_Data("../External/Data/fb-net1.csv")
Load_Rules("../External/Rules/Pi4k.gap")

str_code = com._Create_CommandString(comp.Compile())

Run_FixPoint()

#sys.stdout.write("GAP+:> ")
#while True:
#    sys.stdout.write("GAP+:>")
#    command = sys.stdin.readline()
#    try:
#        exec(command)
#    except Exception as e:
#        print("## Exception :")
#        print(str(e))
#       print()

#endregion

