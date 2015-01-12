__author__ = "Bar Bokovza"

from enum import Enum
import copy
from functools import cmp_to_key

import numpy as np

import Code.validation as valid


class BlockType( Enum ):
    UNKNOWN_BLOCK = 0
    ANNOTATION_BLOCK = 1
    ABOVE_BLOCK = 2

class RuleType( Enum ):
    UNKNOWN = 0
    ONCE_HEADER_RULE = 1
    GROUND_RULE = 2
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


def parse ( rules ) -> list:  # (headerBlock, body_lst)
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


def parse_block ( block ) -> (str, list, str, BlockType):
    """
    Gets a block in a GAP Rule and return a tuple of (atom, args, notation, blockType)
    :param block:
    :return: tuple => (atom:str, args:list, notation:str, blockType:BlockType)
    :raise ValueError:
    """
    predicat, notation = block.split( ":" )
    blockType = BlockType.UNKNOWN_BLOCK
    if valid.IsFloat( notation ):
        blockType = BlockType.ABOVE_BLOCK
    else:
        blockType = BlockType.ANNOTATION_BLOCK

    atoms = predicat.split( "," )
    if len( atoms ) < 2:
        raise ValueError( "The predicat '" + predicat + "' does not have an atom and arguments" )

    atom, args = atoms [0], atoms [1:]

    return atom, args, notation, blockType


def cmp_analyse_rule ( a, b ) -> int:
    a_atom, a_arguments, a_notation, a_type, a_valsPic, a_matches = a
    b_atom, b_arguments, b_notation, b_type, b_valsPic, b_matches = b

    if a_type != b_type:
        return a_type.value - b_type.value

    if len( a_arguments ) < len( b_arguments ):
        return -1
    if len( b_arguments ) < len( a_arguments ):
        return 1

    if len( a_matches ) < len( b_matches ):
        return -1
    if len( b_matches ) < len( a_matches ):
        return 1

    return 0

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
        varPic, matches, = create_varsPic_matches( args, arg_dict )
        finalLst.append( (atom, args, notation, type, varPic, matches) )

    headerRes = finalLst [len( finalLst ) - 1]
    bodyRes = finalLst [0:len( finalLst ) - 1]

    bodyRes = sorted( bodyRes, key = cmp_to_key( cmp_analyse_rule ) )

    return (headerRes, bodyRes)

def cmp_bodyBlock ( a, b ) -> int:
    a_varPic, b_varPic = a [4], b [4]
    a_count, b_count = 0, 0

    for item in a_varPic:
        if item != -1:
            a_count = a_count + 1

    for item in b_varPic:
        if item != -1:
            b_count = a_count + 1

    if a_count > b_count:
        return -1
    if a_count < b_count:
        return 1
    return 0


class GAP_Rule:
    part_header = ()
    part_body = {}
    rule_type = RuleType.UNKNOWN

    def __init__ ( self, header, body ):
        part_header = header
        self.rule_type = RuleType.HEADER_RULE

        if len( body ) > 0:
            self.rule_type = RuleType.GROUND_RULE

            for block in body:
                predicat, args, notation, type = block
                if not type in self.part_body.keys( ):
                    self.part_body [type] = []

                self.part_body [type].append( block )

                if type is BlockType.ABOVE_BLOCK:
                    self.rule_type = RuleType.BASIC_RULE

class GAP_Compiler:
    rules = []

    def __int__ ( self ):
        rules = []

    def load ( self, path ):
        lines = []
        filer = open( path, "r" )

        for line in filer.readlines( ):
            lines.append( line )

        result = parse( lines )

        for rule in result:
            self.rules.append( analyse_rule( rule ) )



