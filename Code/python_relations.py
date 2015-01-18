__author__ = "Bar Bokovza"

#region Imports
import numpy as np
#endregion

#region Private Functions
def Generate_Empty (dtype) -> np.ndarray:
    return np.zeros(0, dtype = dtype)

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

    return result, joinLst

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
class GAP_PythonRelations:
    def Cartesian (self, a:np.ndarray, b:np.ndarray) -> np.ndarray:
        a_row, a_col = np.shape(a)
        b_row, b_col = np.shape(b)

        if a_row == 0 or b_row == 0 or a_col == 0 or b_col == 0:
            return Generate_Empty(np.int32)

        result = np.zeros((a_row * b_row, a_col + b_col), dtype = np.int32)

        for x in range(a_row):
            for y in range(b_row):

                for i in range(a_col):
                    result[x * b_row + y][i] = a[x][i]

                for i in range(b_col):
                    result[x * b_row + y][a_col + i] = a[x][i]

        return result

    def Join (self, a:tuple, b:tuple) -> tuple:
        a_idx, a_valsPic = a
        b_idx, b_valsPic = b

        a_row, a_col = np.shape(a_idx)
        b_row, b_col = np.shape(b_idx)

        join_valsPic, joinLst = Create_VarsPic_Join(a_valsPic, b_valsPic)
        lst_row = np.shape(joinLst)[0]

        if len(joinLst) is 0:
            return self.Cartesian(a_idx, b_idx), join_valsPic

        result = np.zeros((a_row * b_row, a_col + b_col - lst_row), dtype = np.int32)
        current = 0

        # JOIN

        for x in range(a_row):
            for y in range(b_row):

                is_ok = True
                i = 0

                while i < len(joinLst) and is_ok:
                    joinVal = joinLst[i]

                    if a_idx[x][a_valsPic[joinVal]] != b_idx[y][b_valsPic[joinVal]]:
                        is_ok = False
                        break
                    i += 1

                if is_ok:
                    curr = current
                    current += 1

                    for i in range(len(join_valsPic)):
                        if join_valsPic[i] != -1:

                            if a_valsPic[i] != -1:
                                result[current][join_valsPic[i]] = a_idx[x][a_valsPic[i]]
                            else:
                                result[current][join_valsPic[i]] = b_idx[x][b_valsPic[i]]

        result = np.resize(result, (current, a_col + b_col - lst_row))
        return result, join_valsPic

    def SelectAbove (self, data:tuple, minValue:float) -> tuple:  # (idx, values)
        a_idx, a_values = data
        a_row, a_col = np.shape(a_idx)

        result_idx = np.zeros(np.shape(a_idx), dtype = np.int32)

        current = 0

        for i in range(a_row):
            if a_values[i] >= minValue:
                for j in range(a_col):
                    result_idx[current][j] = a_idx[i][j]

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

        places = np.array(projectionLst, dtype = np.int32)
        result = np.zeros((data_row, len(projectionLst)), dtype = np.int32)

        for i in range(data_row):
            count = 0
            for j in projectionLst:
                result[i][count] = data[i][j]
                count += 1

        return result

    def Distinct (self, array:np.ndarray, dictionary:dict) -> (np.ndarray, np.ndarray):
        dist = { }

        for item in array:
            tup = tuple(item)
            if not tup in dist.keys() and tup in dictionary.keys():
                dist[tup] = dictionary[tup]

        if len(dist) == 0:
            return np.zeros(0, dtype = np.int32), np.zeros(0, dtype = np.float)

        idx = np.array(list(dist.keys()), dtype = np.int32)
        values = np.array(list(dist.values()), dtype = np.float)

        return idx, values

    def SetLowerBoundary (self, data:tuple, minValue:float) -> tuple:
        a_idx, a_values = data

        a_row, = np.shape(a_values)

        for i in a_row:
            if a_values[i] < minValue:
                a_values[i] = minValue

        return a_idx, a_values

    def SelectAbove_Full (self, a:tuple, virtual_places:list, data:dict, minValue:float, toJoin:bool = False) -> (
            np.ndarray, np.ndarray):
        a_array, a_valsPic = a
        physical_places, places_valsPic = Create_VarsPic_Virtual(a_valsPic, virtual_places), Create_VarsPic_Physical(
            virtual_places, np.shape(a_valsPic)[0])

        projection_array = self.Projection(a_array, physical_places)
        distinct_array = self.Distinct(projection_array, data)
        if np.shape(distinct_array[0])[0] == 0:
            return np.zeros(0, dtype = np.int32), np.zeros(0, dtype = np.float)
        select_idx = self.SelectAbove(distinct_array, minValue)

        if toJoin:
            return self.Join(a, (select_idx, places_valsPic))
        return select_idx, places_valsPic

#endregion

