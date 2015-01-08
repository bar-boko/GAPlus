__author__ = "Bar Bokovza"

from enum import Enum


class BlockType( Enum ):
    ANNOTATION_BLOCK = 1
    ABOVE_BLOCK = 2


class RuleType( Enum ):
    ONCE_HEADER_RULE = 1
    ONCE_GROUND_RULE = 2
    HEADER_RULE = 3
    BASIC_RULE = 4
    COMPLEX_RULE = 5


def compile ( rules ):
    final_results = []

    for rule in rules:
        rule = rule.replace( " ", "" )
        rule = rule.replace( "(", "," )
        rule = rule.replace( ")", "," )
        rule = rule.replace( "[", "," )
        rule = rule.replace( "]", "," )
        str_header, str_body = rule.split( "<-" )


# TESTING
compile( ["g1(x):a<-g2(x):a", "friend(x,y):a*b<-p(x):a&q(y):b"] )
