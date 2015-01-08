__author__ = "Bar Bokovza"

from enum import Enum
import validation
import numpy as np
import copy

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


def create_varsPic_matches ( args, varsDict ) -> tuple:  # (result, matches)
    matches = []
    size = len( varsDict.keys( ) )
    result = np.zeros( size, dtype = np.int32 )

    for i in range( 0, size ):
        result [i] = -1

    count = 0
    for arg in args:
        if not arg in varsDict.keys( ):
            raise ValueError( "The wrong dictionary have been sent to the function." )

        idx = varsDict [arg]

        if result [idx] == -1:
            result [idx] = count
        else:
            matches.append( (result [idx], count) )

        count = count + 1

    return result, matches


def parse ( rules ) -> tuple:  # (headerBlock, body_lst)
    result = []
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

        result.append( (headerBlock, body_lst) )

    return result


def parse_block ( block ) -> tuple:  # (atom, args, notation, blockType)
    predicat, notation = block.split( ":" )
    blockType = BlockType.UNKNOWN_BLOCK
    if validation.v_IsFloat( notation ):
        blockType = BlockType.ABOVE_BLOCK
    else:
        blockType = BlockType.ANNOTATION_BLOCK

    atoms = predicat.split( "," )
    if len( atoms ) < 2:
        raise ValueError( "The predicat '" + predicat + "' does not have an atom and arguments" )

    atom, args = atoms [0], atoms [1:]

    return atom, args, notation, blockType


def analyse_rule ( rule ):
    arg_dict = {}
    predicats = []
    count = 0

    headerBlock, bodyBlocks = rule

    blockLst = copy.deepcopy( bodyBlocks )
    blockLst.append( headerBlock )

    finalLst = []

    for block in blockLst:
        atom, args, notation, type = block
        if not atom in predicats:
            predicats.append( atom )
        for arg in args:
            if not arg in arg_dict:
                arg_dict [arg] = count
                count = count + 1

    for block in blockLst:
        atom, args, notation, type = block
        varPic, matches = create_varsPic_matches( args, arg_dict )
        finalLst.append( (atom, args, notation, type, varPic, matches) )

    headerRes = finalLst [len( finalLst ) - 1]
    bodyRes = finalLst [0:len( finalLst ) - 2]

    return (headerRes, bodyRes)

# TESTING
rule = parse( ["g1_member(X):a*b*c*d<-g1_member(Y):b&p(Y):c&friend(Y,X):a&p(X):d&p(Y):0.25"] )
result = analyse_rule( rule [0] )