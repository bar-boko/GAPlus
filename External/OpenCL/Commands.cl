#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

__kernel
void CARTESIAN(__global const int* a, int a_col, __global const int* a_varsPic,
               __global const int* b, int b_col, __global const int* b_varsPic,
               __global int* target, __global const int* join_varsPic)
{
    int x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
    int rowPosition = x * b_col + y, i;
    
    if (join_varsPic[z] != -1)
    {
        if(a_varsPic[z] != -1)
            target[rowPosition * (a_col + b_col) + join_varsPic[z]] = a[x * a_col + a_varsPic[z]];
        else
            target[rowPosition * (a_col + b_col) + join_varsPic[z]] = b[x * b_col + b_varsPic[z]];
    }
}

__kernel
void SUPER_JOIN(__global const int* a, const int a_col, __global const int* a_varsPic,
          __global const int* b, const int b_col, __global const int* b_varsPic,
          __global const int* joinLst, const int joinLst_length,
          __global const int* join_varsPic, const int varsPic_size,
          __global int* result, __global int* current)
{
    int x = get_global_id(0), y = get_global_id(1);
    int array_size = a_col + b_col - joinLst_length;
        
    int isOk = 1, i = 0;
    for (i = 0;i < joinLst_length; i++) {
        int join = joinLst[i];
        if (a[x*a_col + a_varsPic[join]] != b[y*b_col+b_varsPic[join]])
        {
            isOk = 0;
        }
    }
    
    if (isOk == 1){
        int curr  = atomic_add(current, 1), count = 0;
        
        for(i = 0; i < varsPic_size; i++){
            if (join_varsPic[i] >= 0)
            {
                if(a_varsPic[i] >= 0)
                    result[curr*array_size + join_varsPic[i]] = a[x*a_col + a_varsPic[i]];
                else
                    result[curr*array_size + join_varsPic[i]] = b[y*b_col + b_varsPic[i]];
                count++;
            }
        }
    }
    
}

/// this function return all the rows from a table that have same value for 2 variables
__kernel
void SIMPLE_FILTER(__global const int* a, __global const float* a_vals, int a_col,
            int idx1, int idx2, __global int* buffer, __global float* buffer_vals, __global int* current)
{
    int x = get_global_id(0);
    
    if(a[x*a_col + idx1] == a[x*a_col + idx2])
    {
        int curr = atom_add(current, 1);
        
        buffer_vals[curr] = a_vals[curr];
        
        int i;
        for(i=0; i<a_col; i++)
            buffer[curr*a_col + i] = a[i];
    }
}

int ToFilter(__global int* a, const int a_col, __global int* places, const int places_row, int x)
{
    for(int i = 0; i < places_row; i++) {
        if (a[x*a_col + places[2*i]] != a[x*a_col + places[2*i+1]])
            return 0;
    }
    return 1;
}

__kernel
void FILTER(__global int* a, const int a_col, __global int* places, const int places_row,
            __global int* varsPic, const int varsPic_row, __global int* current, __global int* result, const int result_col) {
                
            int x = get_global_id(0), i, isOk = 1;
            
            if(ToFilter(a, a_col, places, places_row, x) == 1) {
                int curr = atomic_add(current,1);
                
                for(i = 0; i < varsPic_row; i++) {
                    if(varsPic[i] != -1) {
                        result[curr*result_col + varsPic[i]] = a[x*a_col + varsPic[i]];
                    }
                }
            }
}        

/// This function return all the rows from the table that have value >= minVal
__kernel
void SELECT_ABOVE(__global const int* args, __global const float* values, int a_col, int minVal,
                  __global int* buffer_args, __global int* current)
{
    int x = get_global_id(0);
    
    if(values[x] >= minVal)
    {
        int curr = atom_add(current, 1);

        int i;
        for(int i=0; i < a_col; i++)
            buffer_args[curr*a_col+i] = args[x*a_col+i];
    }
}
// This function make sure to have all values >= minValue
__kernel
void SET_LOWER_BOUNDARY(__global int* values, int minValue)
{
    int i = get_global_id(0);
    if(values[i] < minValue)
        values[i] = minValue;
}

__kernel
void PROJECTION(__global const int* array, const int array_col, __global int* result, const int result_col,
                __global const int* places) {
                
                int x = get_global_id(0), y = get_global_id(1);
                result[x*result_col+y] = array[x*array_col + places[y]];
}

