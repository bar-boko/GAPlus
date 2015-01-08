__author__ = "Bar Bokovza"


def v_IsFloat ( number ) -> bool:
    try:
        float( number )
    except:
        return False
    return True


