__author__ = "Bar Bokovza"


def IsFloat ( number ) -> bool:
    try:
        float( number )
    except:
        return False
    return True


