__author__ = "Bar Bokovza"

#region IMPORTS
#import numpy as np
from Code.dataHolder import GAP_Data
#from Code.opencl import GAP_OpenCL
from Code.basic import GAP_Basic
import Code.compiler as com
from time import time
import gc
#endregion

path_data, path_rules = "External/Data/fb-net1.csv", "External/Rules/Pi4a.gap"

data = GAP_Data()
data.Load(path_data)

comp = com.GAP_Compiler()
comp.Load(path_rules)
comp.PreRun()

runner = GAP_Basic()

count, max = 0, 100

print("# BEGIN")
while count <= max:
    rule = comp.Rules[0]
    start = time()
    def_zone = rule.Create_DefinitionZone(data, runner)
    end = time()

    if count == 0:
        gc.collect()

    print(end - start)
    count += 1

print("# END")



