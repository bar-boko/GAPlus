# GAPlus

A Compiler for "General Annotated Logic Programming", using OpenCL for parallel running

This is the 1st version of my compiler that uses OpenCL as part of the GAP Rules running process.
To use the compiler, run the "pygaplus.py" script and follow the instructions.

This compiler uses a special technique :
"GAP Definition Zone" (In OpenCL) - An algorithm that uses OpenCL to minimize the scope of data for each rule, by
taking each GAP Rule and turn it into an "SQL Query" like.


