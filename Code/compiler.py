__author__ = "Bar Bokovza"

from enum import Enum
from code import validation
import numpy as np

class BlockType( Enum ):
    UNKNOWN_BLOCK = 0
    ANNOTATION_BLOCK = 1
    ABOVE_BLOCK = 2


class RuleType( Enum ):
    ONCE_HEADER_RULE = 1
    ONCE_GROUND_RULE = 2
    HEADER_RULE = 3
    BASIC_RULE = 4
    COMPLEX_RULE = 5


def create_varsPic_matches ( args, varsDict ) -> tuple:
    matches = []
    size = len( varsDict.keys( ) )
    result = np.zeros( size, dtype = np.int32 )
    for i in range( 0, size - 1 ):
        result [i] = -1

    count = 0
    for arg in args:
        if not arg in varsDict.keys( ):
            raise ValueError( "The wrong dictionary have been sent to the function." )

        idx = varsDict [arg]

        if result [idx] == -1:
            result [idx] = count
        else:
            matches.append( result [idx], count )

        count = count + 1

    return result, matches


def parse ( rules ) -> tuple:
    final_results = []

    for rule in rules:
        rule = rule.replace( " ", "" )
        rule = rule.replace( "(", "," )
        rule = rule.replace( ")", "" )
        rule = rule.replace( "[", "," )
        rule = rule.replace( "]", "" )
        str_header, str_body = rule.split( "<-" )

        headerBlock = parse_block( str_header )
        if headerBlock [3] != BlockType.ANNOTATION_BLOCK:
            raise ValueError( "In Header the block must be an annotation block." )

        body_lst = []

        for block in str_body.split( "&" ):
            parsedBlock = parse_block( block )
            body_lst.append( parsedBlock )

        final_results.append( (headerBlock, body_lst) )


def parse_block ( block ) -> tuple:
    predicat, notation = block.split( ":" )
    blockType = BlockType.UNKNOWN_BLOCK
    if validation.v_IsFloat( notation ):
        blockType = BlockType.ABOVE_BLOCK
    else:
        blockType = BlockType.ANNOTATION_BLOCK

    atoms = [predicat.split( "," )]
    if len( atoms ) < 2:
        raise ValueError( "The predicat '" + predicat + "' does not have an atom and arguments" )

    atom, args = atoms [0], atoms [1:]

    return atom, args, notation, blockType

# def analyse_rule (rule):


# TESTING
parse( ["g1(x):a<-g2(x):a", "friend(x,y):a*b<-p(x):a&q(y):b"] )
