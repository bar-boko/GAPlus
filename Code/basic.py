__author__ = "Bar Bokovza"

#region Imports
import numpy as np
#endregion

#region Private Functions
def Generate_Empty (dtype) -> np.ndarray:
    return np.zeros(0, dtype = dtype)

def Set_Argument (kernel, idx:int, value, dataType = np.int32):
    kernel.set_arg(idx, np.array([value], dtype = dataType))

def Create_VarsPic_Join (a:np.ndarray, b:np.ndarray) -> np.ndarray:
    a_row, b_row = np.shape(a)[0], np.shape(b)[0]

    if a_row != b_row:
        zero = np.zeros(0, dtype = np.int32)
        return zero, zero

    result = np.zeros(a_row, dtype = np.int32)
    result.fill(-1)
    count = 0
    joinLst = []

    for i in range(a_row):
        a_val, b_val = a[i], b[i]
        if a_val == -1 and b_val == -1:
            result[i] = -1
        else:
            result[i] = count
            count = count + 1

        if a_val != -1 and b_val != -1:
            joinLst.append(i)

    return result, np.array(joinLst, dtype = np.int32)

def Length_VarsPic (varsPic:np.ndarray) -> int:
    size = 0

    for item in varsPic:
        if item is not -1:
            size = size + 1

    return size

def Create_VarsPic_Virtual (varsPic:np.ndarray, places:list) -> np.ndarray:
    result = []

    for place in places:
        result.append(varsPic[place])

    return result

def Create_VarsPic_Physical (lst:list, size:int) -> np.ndarray:
    result = np.zeros(size, dtype = np.int32)
    result.fill(-1)

    count = 0

    for ptr in lst:
        if result[ptr] == -1:
            result[ptr] = count
            count = count + 1

    return result

#endregion

#region GAP OpenCL
class GAP_Basic:
    def Cartesian (self, a:tuple, b:tuple, join_varsPic:np.ndarray) -> tuple:
        a_idx, a_varsPic = a
        a_row, a_col = np.shape(a_idx)

        b_idx, b_varsPic = b
        b_row, b_col = np.shape(b_idx)

        size = np.shape(a_varsPic)[0]

        result = np.zeros((a_row * b_row, Length_VarsPic(join_varsPic)), dtype = np.int32)

        for x in range(a_row):
            for y in range(b_row):
                rowPosition = (x * a_col + y) * (a_col + b_col)
                for z in range(size):
                    if join_varsPic[z] != -1:
                        if (a_varsPic[z] != -1):
                            result[rowPosition + join_varsPic[z]] = a[x * a_col + a_varsPic[z]]
                        else:
                            result[rowPosition + join_varsPic[z]] = b[y * b_col + b_varsPic[z]]

        return result, join_varsPic

    def SelectAbove (self, data:tuple, minValue:float) -> tuple:  # (idx, values)
        a_idx, a_values = data
        a_row, a_col = np.shape(a_idx)

        result_idx = np.zeros(np.shape(a_idx), dtype = np.int32)

        current = 0

        for x in range(a_row):
            if a_values[x] > minValue:
                for i in range(a_col):
                    result_idx[current][i] = a_idx[x][i]

                current += 1

        result_idx = np.resize(result_idx, (current, a_col))

        return result_idx

    def Filter (self, a:tuple, matches:list) -> np.ndarray:
        a_idx, a_varsPic = a
        a_row, a_col = np.shape(a_idx)
        varsPic_row = np.shape(a_varsPic)[0]
        varsPic_size = Length_VarsPic(a_varsPic)

        result_idx = np.zeros(a_row * varsPic_size, dtype = np.int32)

        count = 0
        for x in range(a_row):
            isOk = True
            i = 0
            while isOk and i < len(matches):
                if a_idx[x][matches[i][0]] != a_idx[x][matches[i][1]]:
                    isOk = False
                else:
                    i += 1

            if not isOk:
                continue

            curr = count
            count += 1

            for i in range(varsPic_row):
                if a_varsPic[i] is not -1:
                    result_idx[curr][a_varsPic[i]] = a_idx[x][a_varsPic[i]]

        result_idx = np.resize(result_idx, (count, varsPic_size))
        return result_idx, a_varsPic

    def Projection (self, data:np.ndarray, projectionLst:list) -> np.ndarray:
        data_row, data_col = np.shape(data)

        result = np.zeros((data_row, len(projectionLst)), dtype = np.int32)

        for x in range(data_row):
            for y in range(len(projectionLst)):
                result[x][y] = data[x][projectionLst[y]]

        return result

    def SuperJoin (self, a:tuple, b:tuple) -> tuple:
        a_idx, a_varsPic = a
        b_idx, b_varsPic = b

        a_row, a_col = np.shape(a_idx)
        b_row, b_col = np.shape(b_idx)
        size = np.shape(a_varsPic)[0]

        join_varsPic, joinLst = Create_VarsPic_Join(a_varsPic, b_varsPic)

        if len(joinLst) is 0:
            return self.Cartesian(a, b, join_varsPic)

        result_col = a_col + b_col - len(joinLst)
        result = np.zeros((a_row * b_row, result_col), dtype = np.int32)
        current = 0

        for x in range(a_row):
            for y in range(b_row):

                isOk, i = True, 0
                while i < len(joinLst):
                    join = joinLst[i]
                    if a_idx[x][a_varsPic[join]] != b_idx[y][b_varsPic[join]]:
                        isOk = False
                        break
                    i += 1

                if isOk:
                    for i in range(size):
                        if join_varsPic[i] >= 0:
                            if a_varsPic[i] >= 0:
                                result[current][join_varsPic[i]] = a_idx[x][a_varsPic[i]]
                            else:
                                result[current][join_varsPic[i]] = b_idx[y][b_varsPic[i]]
                    current += 1

        result = np.resize(result, (current, a_col + b_col - len(joinLst)))
        return result, join_varsPic

    def Distinct (self, array:np.ndarray, dictionary:dict = None) -> (np.ndarray, np.ndarray):
        dist = { }

        for item in array:
            tup = tuple(item)

            if not tup in dist.keys():
                if dictionary is not None and tup in dictionary.keys():
                    dist[tup] = dictionary[tup]
                else:
                    dist[tup] = 1

        if len(dist) == 0:
            return np.zeros(0, dtype = np.int32), np.zeros(0, dtype = np.float)

        idx = np.array(list(dist.keys()), dtype = np.int32)
        if not dictionary == None:
            values = np.array(list(dist.values()), dtype = np.float)
        else:
            values = None

        return idx, values

    def SelectAbove_Full (self, a:tuple, virtual_places:list, data:dict, minValue:float, toJoin:bool = False) -> (
            np.ndarray, np.ndarray):
        a_array, a_valsPic = a
        physical_places, places_valsPic = Create_VarsPic_Virtual(a_valsPic, virtual_places), Create_VarsPic_Physical(
            virtual_places, np.shape(a_valsPic)[0])

        projection_array = self.Projection(a_array, physical_places)
        distinct_array = self.Distinct(projection_array, data)
        select_idx = self.SelectAbove(distinct_array, minValue)

        if toJoin:
            return self.SuperJoin(a, (select_idx, places_valsPic))
        return select_idx, places_valsPic

#endregion

