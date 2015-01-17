__author__ = "Bar Bokovza"

import numpy as np

import Code.dataHolder as dat
import Code.compiler as comp
import Code.opencl as para

dataHold = dat.GAP_Data()
dataHold.Load("External/Data/fb-net1.csv")
dataHold.Load("External/Data/fb-net2.csv")
dataHold.Load("External/Data/fb-net3.csv")

com = comp.GAP_Compiler()
com.Load("External/Rules/Pi4i.gap")
code = com.Compile()

gpu = para.GAP_OpenCL()

MainDict = dataHold.data

dict_g1_member = MainDict["g1_member"]
array_g1_member = dataHold.Generate_NDArray("g1_member")
dict_p = MainDict["p"]
array_p = dataHold.Generate_NDArray("p")
dict_friend = MainDict["friend"]
array_friend = dataHold.Generate_NDArray("friend")
dict_q = MainDict["q"]
array_q = dataHold.Generate_NDArray("q")
dict_g2_member = MainDict["g2_member"]
array_g2_member = dataHold.Generate_NDArray("g2_member")

def DefinitionZone_0 () -> tuple:
    start_block_varsPic_0 = np.array([-1, 0], dtype = np.int32)
    start_block_0 = (array_g1_member, start_block_varsPic_0)
    start_block_varsPic_1 = np.array([-1, 0], dtype = np.int32)
    start_block_1 = (array_p, start_block_varsPic_1)
    start_block_varsPic_2 = np.array([0, -1], dtype = np.int32)
    start_block_2 = (array_p, start_block_varsPic_2)
    start_block_varsPic_3 = np.array([0, -1], dtype = np.int32)
    start_block_3 = gpu.Filter((array_friend, start_block_varsPic_3), [(0, 1)])
    size = np.shape(start_block_3[0])[0]
    if size is 0:
        return np.zeros((0, 0), dtype = np.int32)
    start_block_varsPic_4 = np.array([-1, 0], dtype = np.int32)
    start_block_4 = (array_p, start_block_varsPic_4)
    join_0_0 = gpu.Join(start_block_0, start_block_1)
    size = np.shape(join_0_0)[0]
    if size is 0:
        return join_0_0
    join_0_1 = gpu.Join(start_block_2, start_block_3)
    size = np.shape(join_0_1[0])[0]
    if size is 0:
        return join_0_1
    join_0_2 = start_block_4
    join_1_0 = gpu.Join(join_0_0, join_0_1)
    size = np.shape(join_1_0)[0]
    if size is 0:
        return join_1_0
    join_1_1 = join_0_2
    join_2_0 = gpu.Join(join_1_0, join_1_1)
    size = np.shape(join_2_0)[0]
    if size is 0:
        return join_2_0
    select_0 = gpu.SelectAbove_Full(join_2_0, [1], dict_p, 0.5)
    size = np.shape(select_0)
    if size[0] is 0:
        return select_0
    select_join_0_0 = select_0
    return gpu.Join(select_join_0_0, join_2_0)

def Rule_0 (assigns:np.ndarray, varsPic:np.ndarray) -> tuple:
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        a_1 = row[varsPic[1]]
        b = dict_g1_member[(a_1,)]
        c = dict_p[(a_1,)]
        d = dict_p[(a_0,)]
        a = dict_friend[(a_0, a_0,)]
        if (a_0,) not in dict_g1_member.keys() and a * b * c * d > 0:
            added += 1
            dict_g1_member[(a_0,)] = a * b * c * d
        elif a * b * c * d >= dict_g1_member[(a_0,)] + (1e-05):
            changed += 1
            dict_g1_member[(a_0,)] = a * b * c * d
    return added, changed

def DefinitionZone_1 () -> tuple:
    start_block_varsPic_0 = np.array([-1, 0], dtype = np.int32)
    start_block_0 = (array_g1_member, start_block_varsPic_0)
    start_block_varsPic_1 = np.array([-1, 0], dtype = np.int32)
    start_block_1 = (array_p, start_block_varsPic_1)
    start_block_varsPic_2 = np.array([0, -1], dtype = np.int32)
    start_block_2 = (array_q, start_block_varsPic_2)
    start_block_varsPic_3 = np.array([0, -1], dtype = np.int32)
    start_block_3 = gpu.Filter((array_friend, start_block_varsPic_3), [(0, 1)])
    size = np.shape(start_block_3)[0]
    if size is 0:
        return np.zeros((0, 0), dtype = np.int32)
    start_block_varsPic_4 = np.array([-1, 0], dtype = np.int32)
    start_block_4 = (array_p, start_block_varsPic_4)
    join_0_0 = gpu.Join(start_block_0, start_block_1)
    size = np.shape(join_0_0)[0]
    if size is 0:
        return join_0_0
    join_0_1 = gpu.Join(start_block_2, start_block_3)
    size = np.shape(join_0_1)[0]
    if size is 0:
        return join_0_1
    join_0_2 = start_block_4
    join_1_0 = gpu.Join(join_0_0, join_0_1)
    size = np.shape(join_1_0)[0]
    if size is 0:
        return join_1_0
    join_1_1 = join_0_2
    join_2_0 = gpu.Join(join_1_0, join_1_1)
    size = np.shape(join_2_0)[0]
    if size is 0:
        return join_2_0
    select_0 = gpu.SelectAbove_Full(join_2_0, [1], dict_p, 0.25)
    size = np.shape(select_0)
    if size[0] is 0:
        return select_0
    select_join_0_0 = select_0
    return gpu.Join(select_join_0_0, join_2_0)

def Rule_1 (assigns:np.ndarray, varsPic:np.ndarray) -> tuple:
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        a_1 = row[varsPic[1]]
        b = dict_g1_member[(a_1,)]
        c = dict_p[(a_1,)]
        d = dict_q[(a_0,)]
        a = dict_friend[(a_0, a_0,)]
        if (a_0,) not in dict_g1_member.keys() and a * b * c * d > 0:
            added += 1
            dict_g1_member[(a_0,)] = a * b * c * d
        elif a * b * c * d >= dict_g1_member[(a_0,)] + (1e-05):
            changed += 1
            dict_g1_member[(a_0,)] = a * b * c * d
    return added, changed

def DefinitionZone_2 () -> tuple:
    start_block_varsPic_0 = np.array([-1, 0], dtype = np.int32)
    start_block_0 = (array_g2_member, start_block_varsPic_0)
    start_block_varsPic_1 = np.array([-1, 0], dtype = np.int32)
    start_block_1 = (array_q, start_block_varsPic_1)
    start_block_varsPic_2 = np.array([0, -1], dtype = np.int32)
    start_block_2 = (array_p, start_block_varsPic_2)
    start_block_varsPic_3 = np.array([1, 0], dtype = np.int32)
    start_block_3 = (array_friend, start_block_varsPic_3)
    start_block_varsPic_4 = np.array([-1, 0], dtype = np.int32)
    start_block_4 = (array_q, start_block_varsPic_4)
    join_0_0 = gpu.Join(start_block_0, start_block_1)
    size = np.shape(join_0_0)[0]
    if size is 0:
        return join_0_0
    join_0_1 = gpu.Join(start_block_2, start_block_3)
    size = np.shape(join_0_1)[0]
    if size is 0:
        return join_0_1
    join_0_2 = start_block_4
    join_1_0 = gpu.Join(join_0_0, join_0_1)
    size = np.shape(join_1_0)[0]
    if size is 0:
        return join_1_0
    join_1_1 = join_0_2
    join_2_0 = gpu.Join(join_1_0, join_1_1)
    size = np.shape(join_2_0)[0]
    if size is 0:
        return join_2_0
    select_0 = gpu.SelectAbove_Full(join_2_0, [1], dict_q, 0.25)
    size = np.shape(select_0)
    if size[0] is 0:
        return select_0
    select_join_0_0 = select_0
    return gpu.Join(select_join_0_0, join_2_0)

def Rule_2 (assigns:np.ndarray, varsPic:np.ndarray) -> tuple:
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        a_1 = row[varsPic[1]]
        b = dict_g2_member[(a_1,)]
        c = dict_q[(a_1,)]
        d = dict_p[(a_0,)]
        a = dict_friend[(a_1, a_0,)]
        if (a_0,) not in dict_g2_member.keys() and a * b * c * d > 0:
            added += 1
            dict_g2_member[(a_0,)] = a * b * c * d
        elif a * b * c * d >= dict_g2_member[(a_0,)] + (1e-05):
            changed += 1
            dict_g2_member[(a_0,)] = a * b * c * d
    return added, changed

def DefinitionZone_3 () -> tuple:
    start_block_varsPic_0 = np.array([-1, 0], dtype = np.int32)
    start_block_0 = (array_g2_member, start_block_varsPic_0)
    start_block_varsPic_1 = np.array([-1, 0], dtype = np.int32)
    start_block_1 = (array_q, start_block_varsPic_1)
    start_block_varsPic_2 = np.array([0, -1], dtype = np.int32)
    start_block_2 = (array_q, start_block_varsPic_2)
    start_block_varsPic_3 = np.array([1, 0], dtype = np.int32)
    start_block_3 = (array_friend, start_block_varsPic_3)
    start_block_varsPic_4 = np.array([-1, 0], dtype = np.int32)
    start_block_4 = (array_q, start_block_varsPic_4)
    join_0_0 = gpu.Join(start_block_0, start_block_1)
    size = np.shape(join_0_0)[0]
    if size is 0:
        return join_0_0
    join_0_1 = gpu.Join(start_block_2, start_block_3)
    size = np.shape(join_0_1)[0]
    if size is 0:
        return join_0_1
    join_0_2 = start_block_4
    join_1_0 = gpu.Join(join_0_0, join_0_1)
    size = np.shape(join_1_0)[0]
    if size is 0:
        return join_1_0
    join_1_1 = join_0_2
    join_2_0 = gpu.Join(join_1_0, join_1_1)
    size = np.shape(join_2_0)[0]
    if size is 0:
        return join_2_0
    select_0 = gpu.SelectAbove_Full(join_2_0, [1], dict_q, 0.75)
    size = np.shape(select_0)
    if size[0] is 0:
        return select_0
    select_join_0_0 = select_0
    return gpu.Join(select_join_0_0, join_2_0)

def Rule_3 (assigns:np.ndarray, varsPic:np.ndarray) -> tuple:
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        a_1 = row[varsPic[1]]
        b = dict_g2_member[(a_1,)]
        c = dict_q[(a_1,)]
        d = dict_q[(a_0,)]
        a = dict_friend[(a_1, a_0,)]
        if (a_0,) not in dict_g2_member.keys() and a * b * c * d > 0:
            added += 1
            dict_g2_member[(a_0,)] = a * b * c * d
        elif a * b * c * d >= dict_g2_member[(a_0,)] + (1e-05):
            changed += 1
            dict_g2_member[(a_0,)] = a * b * c * d
    return added, changed

def DefinitionZone_4 () -> tuple:
    start_block_varsPic_0 = np.array([0], dtype = np.int32)
    start_block_0 = (array_g2_member, start_block_varsPic_0)
    start_block_varsPic_1 = np.array([0], dtype = np.int32)
    start_block_1 = (array_p, start_block_varsPic_1)
    start_block_varsPic_2 = np.array([0], dtype = np.int32)
    start_block_2 = (array_q, start_block_varsPic_2)
    start_block_varsPic_3 = np.array([0], dtype = np.int32)
    start_block_3 = (array_p, start_block_varsPic_3)
    join_0_0 = gpu.Join(start_block_0, start_block_1)
    size = np.shape(join_0_0)[0]
    if size is 0:
        return join_0_0
    join_0_1 = gpu.Join(start_block_2, start_block_3)
    size = np.shape(join_0_1)[0]
    if size is 0:
        return join_0_1
    join_1_0 = gpu.Join(join_0_0, join_0_1)
    size = np.shape(join_1_0)[0]
    if size is 0:
        return join_1_0
    select_0 = gpu.SelectAbove_Full(join_1_0, [0], dict_p, 0.25)
    size = np.shape(select_0)
    if size[0] is 0:
        return select_0
    select_join_0_0 = select_0
    return gpu.Join(select_join_0_0, join_1_0)

def Rule_4 (assigns:np.ndarray, varsPic:np.ndarray) -> tuple:
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        b = dict_g2_member[(a_0,)]
        c = dict_p[(a_0,)]
        d = dict_q[(a_0,)]
        if (a_0,) not in dict_g1_member.keys() and b * c * d > 0:
            added += 1
            dict_g1_member[(a_0,)] = b * c * d
        elif b * c * d >= dict_g1_member[(a_0,)] + (1e-05):
            changed += 1
            dict_g1_member[(a_0,)] = b * c * d
    return added, changed

def DefinitionZone_5 () -> tuple:
    start_block_varsPic_0 = np.array([0], dtype = np.int32)
    start_block_0 = (array_g1_member, start_block_varsPic_0)
    start_block_varsPic_1 = np.array([0], dtype = np.int32)
    start_block_1 = (array_q, start_block_varsPic_1)
    start_block_varsPic_2 = np.array([0], dtype = np.int32)
    start_block_2 = (array_p, start_block_varsPic_2)
    start_block_varsPic_3 = np.array([0], dtype = np.int32)
    start_block_3 = (array_q, start_block_varsPic_3)
    join_0_0 = gpu.Join(start_block_0, start_block_1)
    size = np.shape(join_0_0)[0]
    if size is 0:
        return join_0_0
    join_0_1 = gpu.Join(start_block_2, start_block_3)
    size = np.shape(join_0_1)[0]
    if size is 0:
        return join_0_1
    join_1_0 = gpu.Join(join_0_0, join_0_1)
    size = np.shape(join_1_0)[0]
    if size is 0:
        return join_1_0
    select_0 = gpu.SelectAbove_Full(join_1_0, [0], dict_q, 0.25)
    size = np.shape(select_0)
    if size[0] is 0:
        return select_0
    select_join_0_0 = select_0
    return gpu.Join(select_join_0_0, join_1_0)

def Rule_5 (assigns:np.ndarray, varsPic:np.ndarray) -> tuple:
    added, changed = 0, 0
    for row in assigns:
        a_0 = row[varsPic[0]]
        b = dict_g1_member[(a_0,)]
        c = dict_q[(a_0,)]
        d = dict_p[(a_0,)]
        if (a_0,) not in dict_g2_member.keys() and b * c * d > 0:
            added += 1
            dict_g2_member[(a_0,)] = b * c * d
        elif b * c * d >= dict_g2_member[(a_0,)] + (1e-05):
            changed += 1
            dict_g2_member[(a_0,)] = b * c * d
    return added, changed

def_zone_0 = DefinitionZone_0()
def_zone_1 = DefinitionZone_1()
def_zone_2 = DefinitionZone_2()
def_zone_3 = DefinitionZone_3()
def_zone_4 = DefinitionZone_4()
def_zone_5 = DefinitionZone_5()
exit()
