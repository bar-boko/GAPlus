__author__ = "Bar Bokovza"

#region Imports
import numpy as np
#endregion

#region Private Functions
def Generate_Empty (dtype):
    """
    "Create empty numpy array
    :param dtype: the dtype of the array
    :return: empty array
    """
    return np.zeros(0, dtype = dtype)

def Set_Argument (kernel, idx, value, dataType = np.int32):
    """
    Set an primitive type variable as an argument to a kernel
    :param kernel: The kernel to set the argument on
    :param idx: The index of the argument. (index of first argument in the kernel = 0)
    :param value: the value of the variable
    :param dataType: the type of the variable
    :rtype: void
    """
    kernel.set_arg(idx, np.array([value], dtype = dataType))

def Create_VarsPic_Join (a, b):
    """
    Create a joined physical Variable Picture based on 2 Physical Variable Pictures.
    :param a: The physical Variable Picture of the first array
    :type a: np.ndarray
    :param b: The physical Variable Picture of the second array
    :type b: np.ndarray
    :return: Joined Physical Variable Picture
    :rtype: np.ndarray
    """
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

def Length_VarsPic (varsPic):
    """
    Evaluate how many variables are in the array, based on the Physical Variables Picture.
    :param varsPic: Physical Variables Picture of an array.
    :type varsPic: np.ndarray
    :return: amount of variables avaliable based on the Physical Variables Picture.
    :rtype: int
    """
    size = 0

    for item in varsPic:
        if item is not -1:
            size = size + 1

    return size

def Create_VarsPic_Places (varsPic, places):
    """
    Create a list of pysical places based on list "places" and the Physical Variables Picture of an array.
    :param varsPic: A Physical Variables Picture of an array
    :type: varsPic: np.ndarray
    :param places: List of numbers of variables
    :type: list
    :return: List of Physical places in the array.
    :rtype: list
    """
    result = []

    for place in places:
        result.append(varsPic[place])

    return result

def Create_VarsPic_Physical (lst, size):
    """
    Create Physical Variables Picture from a Virtual Variables Picture
    :param lst: A Virtual Variables Picture
    :type lst: list
    :param size: An amount of variables in the rule
    :type size: int
    :return: A Physical Variables Picture
    :rtype: np.ndarray
    """
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
    """
    Implementation of the rational functions in Python
    """

    def Cartesian (self, a, b, join_varsPic):
        """
        Implement Cartesian Multiplication between relations
        :param a: (Array, Physical Variables Pictures of the array)
        :type a:tuple
        :type b:tuple
        :param b: (Array, Physical Variables Pictures of the array)
        :param join_varsPic: Physical Variables Pictures of the demanded array
        :type join_varsPic: np.ndarray
        :return: (Array, join_varsPic)
        :rtype: tuple
        """
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

    def SelectAbove (self, data, minValue):
        """
        Implement Projection[Indexes] { Selection [Value >= minValue] {indexes, values}}
        :param data: (Indexes Array, Values Array)
        :type data: tuple
        :param minValue: the minimum value of items that we demanded
        :type minValue: float
        :return: Indexes Array
        :rtype: np.ndarray
        """
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

    def Filter (self, a, matches):
        """
        Implement Selection [List of (Field1 = Field2) connected with AND] {array}
        :param a: (Array, Physical Variables Picture)
        :type a:tuple
        :param matches: list of matches
        :type matches: list
        :return: (Array, Physical Variables Picture)
        :rtype: tuple
        """
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

    def Projection (self, data, projectionLst):
        """
        Implement Projection [list]
        :param data: (Array)
        :type data: np.ndarray
        :param projectionLst: List of demanded fields
        :type projectionLst:list
        :return: Array of projected Array (+duplicates)
        """
        data_row, data_col = np.shape(data)

        result = np.zeros((data_row, len(projectionLst)), dtype = np.int32)

        for x in range(data_row):
            for y in range(len(projectionLst)):
                result[x][y] = data[x][projectionLst[y]]

        return result

    def SuperJoin (self, a, b):
        """
        Implement Join between two tables.
        :param a: (Array, Physical Variables Picture)
        :type a: tuple
        :param b: (Array, Physical Variables Picture)
        :type b: tuple
        :return: (Joined Array, Joined Physical Variables Picture)
        :rtype: tuple
        """
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

    def Distinct (self, array, dictionary = None):
        """
        Implement Distinct on an array
        :param array: An array
        :type array: np.ndarray
        :param dictionary: [Optional] if dictionary exist, the function will return also the values for the distinct
         entries
        :type dictionary: dict
        :return: (Indexes Array, Values Array)
        :rtype: tuple
        """
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

    def SelectAbove_Full (self, a, virtual_places, data, minValue, toJoin = False):
        """
        Execution the Full process of Select Above - Preparation for Select Above + Select Above
        :param a: (Array, Physical Variables Picture)
        :param virtual_places:
        :param data: The Dictionary to get the data from.
        :param minValue: The minimum value
        :param toJoin: [Optional] if to join to the original array [default = FALSE]
        :return:
        """
        a_array, a_valsPic = a
        physical_places, places_valsPic = Create_VarsPic_Places(a_valsPic, virtual_places), Create_VarsPic_Physical(
            virtual_places, np.shape(a_valsPic)[0])

        projection_array = self.Projection(a_array, physical_places)
        distinct_array = self.Distinct(projection_array, data)
        select_idx = self.SelectAbove(distinct_array, minValue)

        if toJoin:
            return self.SuperJoin(a, (select_idx, places_valsPic))
        return select_idx, places_valsPic

#endregion

