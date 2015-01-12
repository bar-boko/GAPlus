__author__ = "Bar Bokovza"

# region IMPORTS
from enum import Enum

import numpy as np

import Code.validation as valid

# endregion

# region ENUMS
class BlockType( Enum ):
    UNKNOWN = 0
    ANNOTATION = 1
    ABOVE = 2


class RuleType( Enum ):
    UNKNOWN = 0
    ONCE_HEADER = 1
    GROUND = 2
    HEADER = 3
    BASIC = 4
    COMPLEX = 5


# endregion

#region p_ Functions

def _Parse_Block ( block ) -> (str, list, str, BlockType):
    """
    Gets a block in a GAP Rule and return a tuple of (atom, args, notation, blockType)
    :param block:
    :return: tuple => (atom:str, args:list, notation:str, blockType:BlockType)
    :raise ValueError:
    """
    predicat, notation = block.split( ":" )

    atoms = predicat.split( "," )
    if len( atoms ) < 2:
        raise ValueError( "The predicat '" + predicat + "' does not have an atom and arguments" )

    atom, args = atoms [0], atoms [1:]

    return atom, args, notation


def _Parse_Rule ( rule ) -> (tuple, list, list):  # (headerBlock, body_lst)
    args = []

    rule = rule.replace( " ", "" )
    rule = rule.replace( "(", "," )
    rule = rule.replace( ")", "" )
    rule = rule.replace( "[", "," )
    rule = rule.replace( "]", "" )
    str_header, str_body = rule.split( "<-" )

    headerBlock = _Parse_Block( str_header )
    args.append( headerBlock [1] )

    body_lst = []

    for block in str_body.split( "&" ):
        parsedBlock = _Parse_Block( block )
        body_lst.append( parsedBlock )
        args.append( parsedBlock [1] )

    return headerBlock, body_lst, args


def _Create_VirtualVarsPic ( arguments, dict ) -> np.ndarray:
    virtual = []

    for arg in arguments:
        virtual.append( dict [arg] )

    return np.array( virtual, dtype = np.int32 )


def _Create_PhysicalVarsPic ( virtual, size ):
    result = np.zeros( size, dtype = np.int32 )
    result.fill( -1 )

    count = 0
    matches = []

    for ptr in virtual:
        if result [ptr] == -1:
            result [ptr] = count
            count = count + 1
        else:
            matches.append( (result [ptr], count) )

    return result, matches


def _Create_ArgumentsDictionary ( lst ):
    result = {}
    count = 0

    for args in lst:
        for arg in args:
            if not arg in result:
                result [arg] = count
                count = count + 1

    return result


#endregion

#region Query Tree
# The number presents the amount of tabs before

def QueryTree_Create_Dictionaries ( lst ) -> list:
    result = []

    for predicat in lst:
        result.append( (predicat + " = MainDict[\"" + predicat + "\"]", 0) )

    return result


def QueryTree_Create_RuleArgs ( block, num:int ) -> list:
    result = []
    predicat, physic = block.Predicat, block.PhysicalVarsPic

    result.append( ("start_block_valsPic_" + str( num ) + "=np.array(" + str( physic ) + ", dtype=np.int32))", 0) )
    result.append( ("start_block_" + str( num ) + " = (" + predicat + ", start_block_valsPic_" + str( num ) + ")", 0) )
    return result


def QueryTree_Create_Filter ( rule, num:int ) -> list:
    result = []
    predicat, physic, matches = rule.Predicat, rule.PhysicalVarsPic, rule.Matches

    result.append( ("start_block_valsPic_" + str( num ) + "=np.array(" + str( physic ) + ", dtype=np.int32))", 0) )
    result.append( ("start_block_" + str( num ) + " = cl.filter((" + predicat + ", start_block_valsPic_" + \
                    str( num ) + "),np.array(" + str( matches ) + ", dtype=np.int32))", 0) )

    return result


def QueryTree_Create_Join ( lst:list ) -> list:
    joinLst = []
    result = []
    interval = 0
    count = 0

    if len( lst ) is 1:
        result.append( ("join_0_0 = start_block_0", 0) )
        return result

    while len( lst ) > 0:
        a = lst.pop( 0 )
        if len( lst ) > 0:
            b = lst.pop( 0 )
            command = "join_" + str( interval ) + "_" + str( count ) + "=cl.join(start_block_" + str( a )
            command = command + ",start_block_" + str( b ) + ")"
            result.append( (command, 0) )
            joinLst.append( (interval, count) )
            count = count + 1
        else:
            result.append( ("join_" + str( interval ) + "_" + str( count ) + "=start_block_" + str( a )) )
            joinLst.append( (interval, count) )

    while len( joinLst ) is not 1:
        temp = []
        interval = interval + 1
        count = 0

        while len( joinLst ) > 0:
            a = joinLst.pop( 0 )
            if len( lst ) > 0:
                b = lst.pop( 0 )
                command = "join_" + str( interval ) + "_" + str( count ) + "=cl.join(join_" + str( a [0] ) + "_" + str(
                    a [1] )
                command = command + ",join_" + str( b [0] ) + "_" + str( b [1] ) + ")"
                result.append( (command, 0) )
                temp.append( (interval, count) )
                count = count + 1
        else:
            result.append(
                ("join_" + str( interval ) + "_" + str( count ) + "=join_" + str( a [0] ) + "_" + str( a [1] ), 0) )
            temp.append( (interval, count) )

        joinLst = temp

    return result, joinLst [0]


#endregion

#region GAP Block
class GAP_Block:
    def __init__ ( self, parsed:tuple, dictionary:dict ):
        predicat, arguments, notation = parsed
        self.Predicat, self.Notation = predicat, notation

        self.Notation = self.Notation.replace( "\n", "" )
        self.Notation = self.Notation.replace( "\r", "" )

        self.Type = BlockType.ANNOTATION
        if valid.IsFloat( self.Notation ):
            self.Type = BlockType.ABOVE

        self.VirtualVarsPic = _Create_VirtualVarsPic( arguments, dictionary )
        size = len( dictionary )
        self.PhysicalVarsPic, self.Matches = _Create_PhysicalVarsPic( self.VirtualVarsPic, size )

    def Bool_NeedFilter ( self ) -> bool:
        return not len( self.Matches ) is 0

    def __str__ ( self ):
        result = self.Predicat + " ("
        for num in self.VirtualVarsPic:
            result = result + int( num ).__str__( ) + " "

        result = result + "): " + self.Notation
        return result

    def __repr__ ( self ):
        return self.__str__( )

    def p_cmp ( self, other ):
        if self.Type.value is not other.Type.value:
            return self.Type.value - other.Type.value

        if len( self.Matches ) is not len( other.Matches ):
            return len( self.Matches ) - len( other.Matches )

        if np.shape( self.VirtualVarsPic ) is not np.shape( other.VirtualVarsPic ):
            return np.shape( self.VirtualVarsPic ) [0] - np.shape( other.VirtualVarsPic ) [0]

        return 0

    def __eq__ ( self, other ):
        return self.p_cmp( other ) is 0

    def __ge__ ( self, other ):
        return self.p_cmp( other ) >= 0

    def __gt__ ( self, other ):
        return self.p_cmp( other ) > 0

    def __le__ ( self, other ):
        return self.p_cmp( other ) <= 0

    def __lt__ ( self, other ):
        return self.p_cmp( other ) < 0


#endregion

#region GAP Rule
class GAP_Rule:
    def __init__ ( self, rule:str ):
        headerBlock, bodyBlock, args = _Parse_Rule( rule )
        self.Dictionary = _Create_ArgumentsDictionary( args )
        self.Body = []
        self.Header = GAP_Block( headerBlock, self.Dictionary )

        for block in bodyBlock:
            self.Body.append( GAP_Block( block, self.Dictionary ) )

        self.Body.sort( )

    def __str__ ( self ):
        result = ""

        result = result + self.Header.__str__( ) + " <- "
        for i in range( 0, len( self.Body ) - 1 ):
            result = result + self.Body [i].__str__( )
            if i + 1 < len( self.Body ):
                result = result + " & "

        return result

    def __repr__ ( self ):
        return self.__str__( )


#endregion

#region GAP Compiler
class GAP_Compiler:
    def __init__ ( self ):
        self.Rules = []

    def Load ( self, path ):
        filer = open( path, "r" )

        for line in filer.readlines( ):
            rules = GAP_Rule( line )
            self.Rules.append( rules )

#endregion

compiler = GAP_Compiler( )
compiler.Load( "../External/Rules/Pi4m.gap" )

rule = compiler.Rules [0]
result0 = QueryTree_Create_Filter( rule.Body [3], 3 )
