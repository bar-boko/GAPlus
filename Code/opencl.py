__author__ = "Bar Bokovza"

#region Imports
import opencl4py as cl
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
class GAP_OpenCL:
    def __init__ (self, path:str = "External/OpenCL/Commands.cl"):
        platforms = cl.Platforms()
        self.platform = platforms.platforms[0]

        devices = self.platform.devices
        self.device = devices[1]

        self.context = cl.Context(self.platform, [self.device])
        self.queue = self.context.create_queue(self.device)

        filer = open(path, "r")
        txtProgram = filer.read()

        self.program = self.context.create_program(txtProgram)
        self.flag_read = cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_ONLY
        self.flag_write = cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_WRITE_ONLY
        self.flag_both = cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_WRITE

    def Cartesian (self, a:np.ndarray, b:np.ndarray) -> np.ndarray:
        a_row, a_col = np.shape(a)
        b_row, b_col = np.shape(b)

        if a_row == 0 or b_row == 0 or a_col == 0 or b_col == 0:
            return Generate_Empty(np.int32)

        result = np.zeros((a_row * b_row, a_col + b_col), dtype = np.int32)

        buffer_a = self.context.create_buffer(self.flag_read, a)
        buffer_b = self.context.create_buffer(self.flag_read, b)
        buffer_result = self.context.create_buffer(self.flag_write, result)

        kernel = self.program.get_kernel("CARTESIAN")

        kernel.set_arg(0, buffer_a)
        Set_Argument(kernel, 1, a_col, np.int32)

        kernel.set_arg(2, buffer_b)
        Set_Argument(kernel, 3, b_col, np.int32)

        kernel.set_arg(4, buffer_result)

        self.queue.execute_kernel(kernel, (a_row, b_row), None)
        self.queue.read_buffer(buffer_result, result)

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
        current = np.zeros(1, dtype = np.int32)

        # BUFFERS
        buffer_a = self.context.create_buffer(self.flag_read, a_idx)
        buffer_b = self.context.create_buffer(self.flag_read, b_idx)

        buffer_a_pic = self.context.create_buffer(self.flag_read, a_valsPic)
        buffer_b_pic = self.context.create_buffer(self.flag_read, b_valsPic)
        buffer_join_pic = self.context.create_buffer(self.flag_read, join_valsPic)

        buffer_result = self.context.create_buffer(self.flag_write, result)
        buffer_current = self.context.create_buffer(self.flag_both, current)

        kernel = self.program.get_kernel("SIMPLE_JOIN")
        if lst_row > 1:
            kernel = self.program.get_kernel("COMPLEX_JOIN")
            buffer_joinLst = self.context.create_buffer(self.flag_read, joinLst)
            kernel.set_arg(10, buffer_joinLst)
            Set_Argument(kernel, 11, lst_row, np.int32)
        else:
            Set_Argument(kernel, 10, joinLst[0], np.int32)

        kernel.set_arg(0, buffer_a)
        Set_Argument(kernel, 1, a_col, np.int32)
        kernel.set_arg(2, buffer_a_pic)

        kernel.set_arg(3, buffer_b)
        Set_Argument(kernel, 4, b_col, np.int32)
        kernel.set_arg(5, buffer_b_pic)

        kernel.set_arg(6, buffer_join_pic)
        Set_Argument(kernel, 8, np.shape(a_valsPic)[0], np.int32)
        kernel.set_arg(7, buffer_current)
        kernel.set_arg(9, buffer_result)

        self.queue.execute_kernel(kernel, (a_row, b_row), None)
        self.queue.read_buffer(buffer_result, result)
        self.queue.read_buffer(buffer_current, current)

        result = np.resize(result, (current[0], a_col + b_col - lst_row))
        return result, join_valsPic

    def SelectAbove (self, data:tuple, minValue:float) -> tuple:  # (idx, values)
        a_idx, a_values = data
        a_row, a_col = np.shape(a_idx)

        result_idx = np.zeros(np.shape(a_idx), dtype = np.int32)

        current = np.zeros(1, dtype = np.int32)

        buffer_idx = self.context.create_buffer(self.flag_read, a_idx)
        buffer_values = self.context.create_buffer(self.flag_read, a_values)

        buffer_result_idx = self.context.create_buffer(self.flag_write, result_idx)

        buffer_current = self.context.create_buffer(self.flag_both, current)

        kernel = self.program.get_kernel("SELECT_ABOVE")

        kernel.set_arg(0, buffer_idx)
        kernel.set_arg(1, buffer_values)
        Set_Argument(kernel, 2, a_col, np.int32)
        Set_Argument(kernel, 3, minValue, np.int32)

        kernel.set_arg(4, buffer_result_idx)
        kernel.set_arg(5, buffer_current)

        self.queue.execute_kernel(kernel, [a_row], None)
        self.queue.read_buffer(buffer_current, current)
        self.queue.read_buffer(buffer_result_idx, result_idx)

        result_idx = np.resize(result_idx, (current[0], a_col))

        return result_idx

    def Filter (self, a:tuple, matches:list) -> np.ndarray:
        a_idx, a_varsPic = a
        a_row, a_col = np.shape(a_idx)
        varsPic_row = np.shape(a_varsPic)[0]
        varsPic_size = Length_VarsPic(a_varsPic)

        a_array = np.resize(a_idx, a_row * a_col)

        matches_array = np.array(matches, dtype = np.int32)
        #matches_array = np.resize(np.array(matches, dtype = np.int32), 2*len(matches))

        result_idx = np.zeros(a_row * varsPic_size, dtype = np.int32)
        current = np.zeros(1, dtype = np.int32)

        """
        buffer_a = self.context.create_buffer(self.flag_read, a_array)
        buffer_places = self.context.create_buffer(self.flag_read, matches_array)
        buffer_varsPic = self.context.create_buffer(self.flag_read, a_varsPic)

        buffer_result = self.context.create_buffer(self.flag_write, result_idx)
        buffer_current = self.context.create_buffer(self.flag_both, current)

        kernel = self.program.get_kernel("FILTER")

        kernel.set_arg(0, buffer_a)
        Set_Argument(kernel, 1, a_row, np.int32)

        kernel.set_arg(2, buffer_places)
        Set_Argument(kernel, 3, len(matches), np.int32)
        kernel.set_arg(4, buffer_varsPic)
        Set_Argument(kernel, 5, varsPic_row, np.int32)

        kernel.set_arg(6, buffer_current)
        kernel.set_arg(7, buffer_result)
        Set_Argument(kernel, 8, varsPic_size)

        self.queue.execute_kernel(kernel, [a_row], None)
        self.queue.read_buffer(buffer_current, current)
        self.queue.read_buffer(buffer_result, result_idx)


        """
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

        result_idx = np.resize(result_idx, (current[0], a_col))
        #result_idx = np.resize(result_idx, (count, varsPic_size))
        return result_idx, a_varsPic

    def Projection (self, data:np.ndarray, projectionLst:list) -> np.ndarray:
        data_row, data_col = np.shape(data)

        places = np.array(projectionLst, dtype = np.int32)
        result = np.zeros((data_row, len(projectionLst)), dtype = np.int32)

        buffer_data = self.context.create_buffer(self.flag_read, data)
        buffer_result = self.context.create_buffer(self.flag_write, result)
        buffer_places = self.context.create_buffer(self.flag_read, places)

        kernel = self.program.get_kernel("PROJECTION")

        kernel.set_arg(0, buffer_data)
        Set_Argument(kernel, 1, data_col, np.int32)
        kernel.set_arg(2, buffer_result)
        Set_Argument(kernel, 3, len(projectionLst), np.int32)
        kernel.set_arg(4, buffer_places)

        self.queue.execute_kernel(kernel, [data_row, len(projectionLst)], None)
        self.queue.read_buffer(buffer_result, result)

        return result

    def Distinct (self, array:np.ndarray, dictionary:dict) -> (np.ndarray, np.ndarray):
        dist = { }

        for item in array:
            tup = tuple(item)
            if not tup in dist.keys():
                dist[tup] = dictionary[tup]

        if len(dist) == 0:
            return None, None

        idx = np.array(list(dist.keys()), dtype = np.int32)
        values = np.array(list(dist.values()), dtype = np.float)

        return idx, values

    def SetLowerBoundary (self, data:tuple, minValue:float) -> tuple:
        a_idx, a_values = data

        a_row, = np.shape(a_values)

        buffer_values = self.context.create_buffer(self.flag_both, a_values)

        kernel = self.program.get_kernel("SET_LOWER_BOUNDARY")

        kernel.set_arg(0, buffer_values)
        Set_Argument(kernel, 1, minValue, np.int32)

        self.queue.execute_kernel(kernel, (a_row), None)
        self.queue.read_buffer(buffer_values, a_values)

        return a_idx, a_values

    def SelectAbove_Full (self, a:tuple, virtual_places:list, data:dict, minValue:float, toJoin:bool = False) -> (
            np.ndarray, np.ndarray):
        a_array, a_valsPic = a
        physical_places, places_valsPic = Create_VarsPic_Virtual(a_valsPic, virtual_places), Create_VarsPic_Physical(
            virtual_places, np.shape(a_valsPic)[0])

        projection_array = self.Projection(a_array, physical_places)
        distinct_array = self.Distinct(projection_array, data)
        select_idx = self.SelectAbove(distinct_array, minValue)

        if toJoin:
            return self.Join(a, (select_idx, places_valsPic))
        return select_idx, places_valsPic

#endregion

