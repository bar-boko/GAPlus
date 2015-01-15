"""
GAPlus - GAP Parallel Compiler using OpenCL
By Bar Bokovza

Service python code only
"""
__author__ = "Bar Bokovza"

def IsFloat (number) -> bool:
    # noinspection PyBroadException
    try:
        float(number)
    except Exception:
        return False
    return True


