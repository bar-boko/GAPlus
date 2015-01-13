__author__ = 'Bar Bokovza'

import opencl4py as cl
import numpy as np

platforms = cl.Platforms()
platform = platforms.platforms[0]

devices = platform.devices
device = devices[1]

context = cl.Context(platform, [device])
queue = context.create_queue(device)

filer = open("../External/OpenCL/Commands.cl", "r")
txtProgram = filer.read()

program = context.create_program(txtProgram)

flag_read = cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_ONLY
flag_write = cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_WRITE_ONLY
flag_both = cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_WRITE

def set_arg(kernel, idx, value, dtype):
    """
    Setting a simple value as an argument in a given kernel.
    :param kernel: The kernel to add the value to.
    :param idx:  The position (starting from 0) of the argument.
    :param value: The value that you want to add to the kernel as argument.
    :param dtype: The dtype of the value (np.dtype)
    """
    kernel.set_arg(idx, np.array([value], dtype=dtype))

def cartesian(a, b) -> np.ndarray:
    """
    Gets 2 NDArrays and return the multiple of it.
    :rtype : np.ndarray
    :param a: The 1st NDArray that we want to multiple.
    :param b: The 2nd NDArray that we want to multiple.
    :return: An NDArray that is a cartesian relation.
    """
    a_row, a_col = np.shape(a)
    b_row, b_col = np.shape(b)

    if a_row == 0 or b_row == 0 or a_col == 0 or b_col == 0 :
        return np.zeros(0, dtype=np.int32)

    result = np.zeros((a_row*b_row, a_col+b_col), dtype=np.int32)

    buffer_a = context.create_buffer(flag_read, a)
    buffer_b = context.create_buffer(flag_read, b)
    buffer_result = context.create_buffer(flag_write, result)

    kernel = program.get_kernel("CARTESIAN")

    kernel.set_arg(0, buffer_a)
    set_arg(kernel, 1, a_col, np.int32)

    kernel.set_arg(2, buffer_b)
    set_arg(kernel, 3, b_col, np.int32)

    kernel.set_arg(4, buffer_result)

    queue.execute_kernel(kernel, (a_row, b_row), None)
    queue.read_buffer(buffer_result, result)

    return result

def p_make_join_valsPic ( a, b ) -> np.ndarray:
    a_row, b_row = np.shape( a ), np.shape( b )

    if a_row != b_row:
        zero = np.zeros( 0, dtype = np.int32 )
        return zero, zero

    result = np.zeros( a_row, dtype = np.int32 )
    count = 0
    joinLst = []

    for i in range( 0, a_row - 1 ):
        a_val, b_val = a [i], b [i]
        if a_val == -1 and b_val == -1:
            result [i] = -1
        else:
            result [i] = count
            count = count + 1

            if a_val != -1 and b_val != -1:
                joinLst.append( i )

    return result, np.array( joinLst, dtype = np.int32 )

def join ( a, b ) -> tuple:
    a_idx, a_valsPic = a
    b_idx, b_valsPic = b

    a_row, a_col = np.shape( a )
    b_row, b_col = np.shape( b )

    join_valsPic, joinLst = p_make_join_valsPic( a_valsPic, b_valsPic )
    lst_row = np.shape( joinLst )

    if lst_row == 0:
        return cartesian( a_idx, b_idx ), join_valsPic

    result = np.zeros( (a_row * b_row, a_col + b_col - lst_row), dtype = np.int32 )
    current = np.zeros( 1, dtype = np.int32 )

    # BUFFERS
    buffer_a = context.create_buffer( flag_read, a_idx )
    buffer_b = context.create_buffer( flag_read, b_idx )

    buffer_a_pic = context.create_buffer( flag_read, a_valsPic )
    buffer_b_pic = context.create_buffer( flag_read, b_valsPic )
    buffer_join_pic = context.create_buffer( flag_read, join_valsPic )

    buffer_result = context.create_buffer( flag_write, result )
    buffer_current = context.create_buffer( flag_both, current )

    kernel = program.get_kernel( "SIMPLE_JOIN" )
    if lst_row > 1:
        kernel = program.get_kernel( "COMPLEX_JOIN" )
        buffer_joinLst = context.create_buffer( flag_read, joinLst )
        kernel.set_arg( 10, buffer_joinLst )
        set_arg( kernel, 11, lst_row, np.int32 )
    else:
        set_arg( kernel, 10, joinLst [0], np.int32 )

    kernel.set_arg( 0, buffer_a )
    set_arg( kernel, 1, a_col, np.int32 )
    kernel.set_arg( 2, buffer_a_pic )

    kernel.set_arg( 3, buffer_b )
    set_arg( kernel, 4, b_col, np.int32 )
    kernel.set_arg( 5, buffer_b_pic )

    kernel.set_arg( 6, buffer_join_pic )
    set_arg( kernel, 7, np.shape( a_valsPic ), np.int32 )
    kernel.set_arg( 8, buffer_current )
    kernel.set_arg( 9, buffer_result )

    queue.execute_kernel( kernel, (a_row, b_row), None )
    queue.read_buffer( buffer_result, result )
    queue.read_buffer( buffer_current, result )

    result = np.resize( result, (current [0], a_col + b_col - lst_row) )
    return result, join_valsPic

def select_above ( data, minValue ) -> tuple:  # (idx, values)
    a_idx, a_vals = data
    a_row, a_col = a_idx

    result_idx, result_vals = np.zeros( np.shape( a_idx ), dtype = np.int32 ), np.zeros( np.shape( a_vals ),
                                                                                         dtype = np.int32 )
    current = np.zeros( 1, dtype = np.int32 )

    buffer_idx = context.create_buffer( flag_read, a_idx )
    buffer_vals = context.create_buffer( flag_read, a_vals )

    buffer_result_idx = context.create_buffer( flag_write, result_idx )
    buffer_result_vals = context.create_buffer( flag_write, result_vals )

    buffer_current = context.create_buffer( flag_both, current )

    kernel = program.get_kernel( "SELECT_ABOVE" )

    kernel.set_arg( 0, buffer_idx )
    kernel.set_arg( 1, buffer_vals )
    set_arg( kernel, 2, a_col, np.int32 )
    set_arg( kernel, 3, minValue, np.int32 )

    kernel.set_arg( 4, buffer_current )
    kernel.set_arg( 5, buffer_result_idx )
    kernel.set_arg( 6, buffer_result_vals )

    queue.execute_kernel( kernel, [a_row], None )
    queue.read_buffer( buffer_current, current )
    queue.read_buffer( buffer_result_idx, result_idx )
    queue.read_buffer( buffer_result_vals, result_vals )

    result_idx = np.resize( result_idx, (current [0], a_col) )
    result_vals = np.resize( result_vals, current [0] )

    return (result_idx, result_vals)

def p_lenVarsPic ( varsPic ) -> int:
    size = 0

    for item in varsPic:
        if item in -1:
            size = size + 1

    return size


def filter ( a, matches ) -> np.ndarray:
    a_idx, a_valsPic = a
    a_row, a_col = np.shape( a_idx )
    valsPic_row = np.shape( a_valsPic )
    valsPic_size = p_lenVarsPic( a_valsPic )

    matches_array = np.array( matches, dtype = np.int32 )

    result_idx = np.zeros( (a_row, valsPic_size), dtype = np.int32 )
    current = np.zeros( 1, dtype = np.int32 )

    buffer_a = context.create_buffer( flag_read, a_idx )
    buffer_places = context.create_buffer( flag_read, matches_array )
    buffer_valsPic = context.create_buffer( flag_read, a_valsPic )

    buffer_result = context.create_buffer( flag_write, result_idx )
    buffer_current = context.create_buffer( flag_both, current )

    kernel = program.get_kernel( "FILTER2" )

    kernel.set_arg( 0, buffer_a )
    set_arg( kernel, 1, a_row, np.int32 )

    kernel.set_arg( 2, buffer_places )
    set_arg( kernel, 3, len( matches ), np.int32 )
    kernel.set_arg( 4, buffer_valsPic )
    set_arg( kernel, 5, valsPic_row, np.int32 )

    kernel.set_arg( 6, buffer_current )
    kernel.set_arg( 7, buffer_result )
    set_arg( kernel, 8, valsPic_size )

    queue.execute_kernel( kernel, [a_row], None )
    queue.read_buffer( buffer_current, current )
    queue.read_buffer( buffer_result, result_idx )

    result_idx = np.resize( result_idx, (current [0], a_col) )
    return result_idx, a_valsPic

def projection ( data, projectionLst ) -> np.ndarray:
    data_row, data_col = np.shape( data )

    places = np.array( projectionLst, dtype = np.int32 )
    result = np.zeros( (data_row, len( projectionLst )), dtype = np.int32 )

    buffer_data = context.create_buffer( flag_read, data )
    buffer_result = context.create_buffer( flag_write, result )
    buffer_places = context.create_buffer( flag_read, places )

    kernel = program.get_kernel( "PROJECTION" )

    kernel.set_arg( 0, buffer_data )
    set_arg( kernel, 1, data_col, dtype = np.int32 )
    kernel.set_arg( 2, buffer_result )
    set_arg( kernel, 3, len( projectionLst ), dtype = np.int32 )
    kernel.set_arg( 4, buffer_places )

    queue.execute_kernel( kernel, [a_row, len( projectionLst )], None )
    queue.read_buffer( buffer_result, result_idx )

    return result

def distinct ( array, dict ) -> (np.ndarray, np.ndarray):
    dist = {}

    for item in array:
        tup = tuple( item )
        if not item in dist.keys( ):
            dist [tup] = dict [tup]

    if len( dist ) == 0:
        return None, None

    idx = np.array( dist.keys( ), dtype = np.int )
    values = np.array( dist.values( ), dtype = np.int )

    return idx, values

def set_lower_boundary ( data, minValue ) -> tuple:
    a_idx, a_values = data

    a_row, = np.shape( a_values )

    buffer_values = context.create_buffer( flag_both, a_values )

    kernel = program.get_kernel( "SET_LOWER_BOUNDARY" )

    kernel.set_arg( 0, buffer_values )
    set_arg( kernel, 1, minValue, np.int32 )

    queue.execute_kernel( kernel, (a_row), None )
    queue.read_buffer( buffer_values, a_values )

    return a_idx, a_values


def p_get_virtual_valsPic ( valuePic, places ) -> np.ndarray:
    result = []

    for place in places:
        result.append( valuePic [place] )

    return result


def p_make_valsPic ( lst, size ) -> np.ndarray:
    result = np.zeros( size, dtype = np.int32 )
    result.fill( -1 )

    count = 0

    for ptr in lst:
        if result [ptr] != -1:
            result [ptr] = count
            count = count + 1

    return result


def full_select_above ( a:tuple, virtual_places:list, data:dict, minValue:float, toJoin:bool = False ) -> (
        np.ndarray, np.ndarray):
    a_array, a_valsPic = a
    physical_places, places_valsPic = p_get_virtual_valsPic( a_valsPic, virtual_places ), p_make_valsPic(
        virtual_places, np.shape( a_valsPic ) )

    projection_array = projection( a_array, physical_places )
    distinct_array = distinct( projection_array, data )
    select_idx, select_vals = select_above( distinct_array, minValue )

    if toJoin:
        return join( a, (select_idx, places_valsPic) )
    return select_idx, valsPic









