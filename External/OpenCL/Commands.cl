#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

/// This function make cartesian multiplication for 2 tables. DONE
__kernel
void CARTESIAN(__global const int* a, int a_col,
               __global const int* b, int b_col,
               __global int* target)
{
    int x = get_global_id(0), y = get_global_id(1);
    int rowPosition = x * b_col + y, i;

    for (i = 0; i < a_col; i++)
        target[rowPosition * (a_col + b_col) + i] = a[x * a_col + i];
    for (i = 0; i < b_col; i++)
        target[rowPosition * (a_col + b_col) + a_col + i] = b[y * b_col + i];
}

/// private function
void EXECUTE_JOIN(__global const int* a, int a_col, __global const int* a_valuesPic,
    	         __global const int* b, int b_col, __global const int* b_valuesPic,
                __global const int* join_valuesPic, __global int* current, const int maxVars, __global int* buffer, int x, int y)
{
    int curr = atomic_add(current, 1);
    
    int i;
    for(i = 0; i < maxVars; i++)
    {
        int writeIdx = curr*(a_col+b_col-1) + join_valuesPic[i], idxX = x*a_col + a_valuesPic[i], idxY = y*b_col + b_valuesPic[i];
        if(a_valuesPic[i] >= 0)
            buffer[writeIdx] = a[idxX];
        else
            buffer[writeIdx] = b[idxY];
    }
}

// This function do a simple join based on 1 variable
__kernel
void SIMPLE_JOIN(__global const int* a, int a_col, __global const int* a_valuesPic,
    	         __global const int* b, int b_col, __global const int* b_valuesPic,
                __global const int* join_valuesPic, __global int* current, const int maxVars, __global int* buffer,
                int joinVar)
{
    int x = get_global_id(0), y = get_global_id(1);
    int rightX = x*a_col + a_valuesPic[joinVar], rightY = y*b_col + b_valuesPic[joinVar];
    
    if(a[rightX] == b[rightY])
        EXECUTE_JOIN(a, a_col, a_valuesPic, b, b_col, b_valuesPic, join_valuesPic, current, maxVars, buffer, x, y);
    /*{
    int curr = atomic_add(current, 1);
    
    int i;
    for(i = 0; i < maxVars; i++)
    {
        int writeIdx = curr*(a_col+b_col-1) + join_valuesPic[i], idxX = x*a_col + a_valuesPic[i], idxY = y*b_col + b_valuesPic[i];
        if(a_valuesPic[i] >= 0)
            buffer[writeIdx] = a[idxX];
        else
            buffer[writeIdx] = b[idxY];
    }
    }*/

}

/// this function do join based on more than 1 variable.
__kernel
void COMPLEX_JOIN(__global const int* a, int a_col, __global const int* a_valuesPic,
                __global const int* b, int b_col, __global const int* b_valuesPic,
                __global const int* join_valuesPic, __global int* current, const int maxVars, __global int* buffer,
                 __global const int* join_a, const int join_row)
{
    int x = get_global_id(0), y = get_global_id(1);
    int join_val, rightX, rightY, i;
    
    for(i=0; i < join_row; i++)
    {
        join_val = join_a[i];
        rightX = x*a_col + a_valuesPic[join_val], rightY = y*b_col + b_valuesPic[join_val];
        if(rightX != rightY)
            return;
    }
    
    //EXECUTE_JOIN(a, a_col, a_row, a_valuesPic, b, b_col, b_row, b_valuesPic, join_valuesPic, current, maxVars, buffer, x, y);
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
void SELECT_ABOVE(__global const int* args, __global const int* values, int a_col, int minVal,
                  __global int* buffer_args, __global int* buffer_values, __global int* current)
{
    int x = get_global_id(0);
    
    if(values[x] >= minVal)
    {
        int curr = atom_add(current, 1);
        buffer_values[curr] = values[x];
        
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

