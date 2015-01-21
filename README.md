# GAPlus
A Compiler for "General Annotated Logic Programming", using OpenCL for parallel running

This is the 1st version of my compiler that uses OpenCL as part of the GAP Rules running process.
To use the compiler, run the "pygaplus.py" script and follow the instructions.

This compiler uses 2 techniques :
"GAP Definition Zone" (In OpenCL) - A complex algorithm that uses OpenCL to minimize the scope of data for each rule, by
taking each GAP Rule and turn it into an "SQL Query" like.
"GAP Flow Engine" - Engine that knows when to execute a GAP Rule by the history of the previous interval(=single running of all rules in the code).

Further details will be avaliable soon.


