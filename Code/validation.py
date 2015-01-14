"""
GAPlus - GAP Parallel Compiler using OpenCL
By Bar Bokovza

Service python code only
"""
__author__ = "Bar Bokovza"


def IsFloat ( number ) -> bool:
    try:
        float( number )
    except:
        return False
    return True


