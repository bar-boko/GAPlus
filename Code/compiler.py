"""
GAPlus - GAP Parallel Compiler using OpenCL
By Bar Bokovza

This is the compiler python file
It get's list of GAP rule, analyse it and return python code to run.
The code that are retrieved are in python, but the CL functions are the OpenCL functions that runs underneath
"""
__author__ = "Bar Bokovza"

# noinspection PyPep8

#region IMPORTS
from enum import Enum

import numpy as np

import Code.validation as valid

#endregion

#region ENUMS
class BlockType(Enum):
    """ Presents the type of block from shape atom(args):notation """
    UNKNOWN = 0
    ANNOTATION = 1
    ABOVE = 2

class RuleType(Enum):
    """
    presents the type of GAP rule
    """
    UNKNOWN = 0
    HEADER = 1
    GROUND = 2
    COMPLEX = 3

#endregion

#region p_ Functions

def _Parse_Block (block) -> (str, list, str, BlockType):
    """
    Gets a block in a GAP Rule and return a tuple of (atom, args, notation, blockType)
    :param block:
    :return: tuple => (atom:str, args:list, notation:str, blockType:BlockType)
    :raise ValueError:
    """
    predicat, notation = block.split(":")

    atoms = predicat.split(",")
    if len(atoms) < 2:
        raise ValueError("The predicat '" + predicat + "' does not have an atom and arguments")

    atom, args = atoms[0], atoms[1:]

    return atom, args, notation

# noinspection PyPep8Naming
def _Parse_Rule (rule:str) -> (tuple, list, list):  # (headerBlock, body_lst)
    """
    gets a string of rule, and return a tuple with header, body and a list of args that are avaliable in the rule
    :param rule: GAP Rule in string
    :return: simplified header, simplified items in body, and list of arguments that are in the rule.
    """
    args = []

    rule = rule.replace(" ", "")
    rule = rule.replace("(", ",")
    rule = rule.replace(")", "")
    rule = rule.replace("[", ",")
    rule = rule.replace("]", "")
    str_header, str_body = rule.split("<-")

    headerBlock = _Parse_Block(str_header)
    args.append(headerBlock[1])

    body_lst = []

    for block in str_body.split("&"):
        parsedBlock = _Parse_Block(block)
        body_lst.append(parsedBlock)
        args.append(parsedBlock[1])

    return headerBlock, body_lst, args

def _Create_VirtualVarsPic (arguments, dictionary) -> np.ndarray:
    """
    transform arg variables from names to numbers
    :param arguments: list of arguments of a block
    :param dictionary: dictionary of arguments
    :return: the list of arguments in numbers instead of names
    """
    virtual = []

    for arg in arguments:
        virtual.append(dictionary[arg])

    return np.array(virtual, dtype = np.int32)

def _Create_PhysicalVarsPic (virtual:np.ndarray, size:int) -> np.ndarray:
    """
    transform virtual pic to physical pic
    :param virtual: the virtual variables picture
    :param size: the amount of unique arguments in rule
    :return:physical variables picture of the virtual that we got as a parameter
    """
    result = np.zeros(size, dtype = np.int32)
    result.fill(-1)

    count = 0
    matches = []

    for ptr in virtual:
        if result[ptr] == -1:
            result[ptr] = count
            count += 1
        else:
            matches.append((result[ptr], count))

    return result, matches

def _Create_ArgumentsDictionary (lst):
    """
    create dictionary of arguments variables based on the list of args that we get from parameter
    :param lst: list of arguments variables in names
    :return: dictionary of that list
    """
    result = { }
    count = 0

    for args in lst:
        for arg in args:
            if not arg in result:
                result[arg] = count
                count += 1

    return result

def _Create_CommandString (lst:list) -> str:
    result = ""

    for line in lst:
        text, tabsCount = line

        command = ""
        for i in range(tabsCount):
            command += "\t"

        command += text
        result += command + "\n"

    return result

#endregion

#region Query Tree
# The number presents the amount of tabs before

def QueryTree_Create_Dictionaries (lst, addon:int = 0) -> list:
    """
    create python code for
    :param lst: list of predicats in the rules
    :param addon: how many tabs to add to the command
    :return: python commands in list and amount of tabs needed
    """
    result = []

    for predicat in lst:
        result.append(("dict_{0}=MainDict[\"{0}\"]".format(predicat), addon))

    return result

def QueryTree_Create_RuleArgs (block, num:int, addon:int = 0) -> list:
    """
    create python code for block that does not need filter
    :param block: GAP Block in GAP_Block form
    :param num: index of the gap block
    :param addon: how many tabs to end to the beginning of the code
    :return: list of python commands
    """
    result = []
    predicat, physic = block.Predicat, block.PhysicalVarsPic

    result.append(("start_block_varsPic_{0}=np.array({1},dtype=np.int32)".format(num, physic.tolist()), addon))
    result.append(("start_block_{0}=({1},start_block_varsPic_{0})".format(num, predicat), addon))

    return result

def QueryTree_Create_Filter (block, num:int, addon:int = 0) -> list:
    """
    create python code for block that needs to pass filtering
    :param block: GAP Block in GAP_Block form
    :param num: index of the gap block
    :param addon: how many tabs to end to the beginning of the code
    :return: list of python commands
    """
    result = []
    predicat, physic, matches = block.Predicat, block.PhysicalVarsPic, block.Matches

    result.append(("start_block_varsPic_{0}=np.array({1},dtype=np.int32)".format(num, physic.tolist()), addon))
    result.append((
        "start_block_{0}=gpu.Filter(({1},start_block_varsPic_{0}), np.array({2},dtype=np.int)".format(num, predicat,
                                                                                                      matches), addon))
    result.append(("_size=np.shape(start_block_{0})[0]".format(num), addon))
    result.append(("if _size is 0:", addon))
    result.append(("return np.zeros((0, 0), dtype=np.int32)", addon + 1))

    return result

def QueryTree_Create_Join (lst:list, in_name:str = "start_block", out_name:str = "join", addon:int = 0) -> list:
    """
    create python code for the join process in the "Definition Zone" paradigm.
    :rtype : list
    :param lst: list of indexes
    :param in_name: name of the starting tables
    :param out_name: name of the output tables
    :param addon: how many tabs to add to the beginning of the code
    :return: list of python commands
    """
    joinLst = []
    result = []
    interval = 0
    count = 0

    if len(lst) is 1:
        result.append((out_name + "_0_0 = " + in_name + "_0", addon))
        return result, [(0, 0)]

    while len(lst) > 0:
        a = lst.pop(0)
        if len(lst) > 0:
            b = lst.pop(0)

            command = "{0}_{1}_{2}=gpu.Join({3}_{4},{3}_{5})".format(out_name, interval, count, in_name, a, b)
            result.append((command, addon))
            command = "_size=np.shape({0}_{1}_{2})[0]".format(out_name, interval, count)
            result.append((command, addon))

            result.append(("if _size is 0:", addon))
            result.append(("return {0}_{1}_{2}".format(out_name, interval, count), addon + 1))
            joinLst.append((interval, count))
            count += 1
        else:
            result.append(("{0}_{1}_{2}={3}_{4}".format(out_name, interval, count, in_name, a), addon))
            joinLst.append((interval, count))

    while len(joinLst) is not 1:
        temp = []
        interval += 1
        count = 0

        while len(joinLst) > 0:
            a = joinLst.pop(0)
            if len(joinLst) > 0:
                b = joinLst.pop(0)

                command = "{0}_{1}_{2}=gpu.Join({0}_{3}_{4},{0},{5},{6})".format(out_name, interval, count, a[0], a[1],
                    b[0], b[1])
                result.append((command, addon))

                result.append(("_size=np.shape({0}_{1}_{2})[0]".format(out_name, interval, count), addon))
                result.append(("if _size is 0:", addon))
                result.append(("return {0}_{1}_{2}".format(out_name, interval, count), addon + 1))

                temp.append((interval, count))
                count += 1
            else:
                result.append(("{0}_{1}_{2}={0}_{3}_{4}".format(out_name, interval, count, a[0], a[1]), addon))
                temp.append((interval, count))

        joinLst = temp

    result.append(("final_{0}={0}_{1}_{2}".format(out_name, joinLst[0][0], joinLst[0][1]), addon))

    return result, joinLst[0]

def QueryTree_Create_SelectAbove (lst:list, rule, dictName:str = "MainDict", in_name:str = "join",
                                  addon:int = 0) -> list:
    """
    creating python code based on the SELECT ABOVE in the "Definition Zone" paradigm.
    :rtype: list
    :param lst: list of indexes
    :param rule: the rule that we check what needs select above
    :param dictName: the dictionary that holds all the data structures
    :param in_name: the input name of the tables
    :param addon: how many tabs we need to add to the beginning of the table
    :return: list of python commands
    """
    result = []
    count = 0

    for ptr in lst:
        command = "select_{0}=gpu.SelectAbove_Full(final_{1},{2},dict_{3},{4})".format(count, in_name,
            rule.Body[ptr].VirtualVarsPic, dictName, rule.Body[ptr].Notation)

        result.append((command, addon))
        result.append(("_size=np.shape(select_{0})".format(count), addon))
        result.append(("if _size[0] is 0:", addon))
        result.append(("return select_{0}".format(count), addon + 1))

    tmp = QueryTree_Create_Join(lst, "select", "select_join", addon = addon)
    joinLst, target = tmp
    result = result + joinLst

    result.append(("return gpu.Join(final_select_join, final_join)", addon))

    return result

# endregion

# region GAP Block
class GAP_Block:
    """
    This is the class of GAP Block
    it analyses GAP block and save all the needed data for it.
    """

    def __init__ (self, parsed:tuple, dictionary:dict):
        predicat, arguments, notation = parsed
        self.Predicat, self.Notation = predicat, notation

        self.Notation = self.Notation.replace("\n", "")
        self.Notation = self.Notation.replace("\r", "")

        self.Type = BlockType.ANNOTATION
        if valid.IsFloat(self.Notation):
            self.Type = BlockType.ABOVE

        self.VirtualVarsPic = _Create_VirtualVarsPic(arguments, dictionary)
        size = len(dictionary)
        self.PhysicalVarsPic, self.Matches = _Create_PhysicalVarsPic(self.VirtualVarsPic, size)

    def Bool_NeedFilter (self) -> bool:
        """

        return true if the arguments in the block needs to pass Filtering
        :return: true or false
        :rtype: bool
        """
        return not len(self.Matches) is 0

    def __str__ (self):
        result = self.Predicat + " ("
        for num in self.VirtualVarsPic:
            result += int(num).__str__() + " "

        result += "): " + self.Notation
        return result

    def __repr__ (self):
        return self.__str__()

    def p_cmp (self, other):
        """
        compare function
        :param other: the other object to compare to
        :return: how much the objects are different
        """
        if self.Type.value is not other.Type.value:
            return self.Type.value - other.Type.value

        if len(self.Matches) is not len(other.Matches):
            return len(self.Matches) - len(other.Matches)

        if np.shape(self.VirtualVarsPic) is not np.shape(other.VirtualVarsPic):
            return np.shape(self.VirtualVarsPic)[0] - np.shape(other.VirtualVarsPic)[0]

        return 0

    def __eq__ (self, other):
        return self.p_cmp(other) is 0

    def __ge__ (self, other):
        return self.p_cmp(other) >= 0

    def __gt__ (self, other):
        return self.p_cmp(other) > 0

    def __le__ (self, other):
        return self.p_cmp(other) <= 0

    def __lt__ (self, other):
        return self.p_cmp(other) < 0

#endregion

#region GAP Rule
class GAP_Rule:
    """
    it analyses and save all the data needed for a single gap rule.
    """

    def __init__ (self, rule:str):
        headerBlock, bodyBlock, args = _Parse_Rule(rule)
        self.Dictionary = _Create_ArgumentsDictionary(args)
        self.Body, self.Predicats = [], []
        self.Header = GAP_Block(headerBlock, self.Dictionary)
        self.Type = RuleType.HEADER

        self.Code_DefinitionZone, self.Code_Run, self.Predicats_Dependent = [], [], []

        self.Predicats.append(headerBlock[0])

        for block in bodyBlock:
            parsed_block = GAP_Block(block, self.Dictionary)

            if parsed_block.Type == BlockType.ANNOTATION and self.Type == RuleType.HEADER:
                self.Type = RuleType.GROUND
            elif parsed_block.Type == BlockType.ABOVE:
                self.Type = RuleType.COMPLEX

            self.Body.append(parsed_block)
            if block[0] not in self.Predicats:
                self.Predicats.append(block[0])
                self.Predicats_Dependent.append(block[0])

        self.Body.sort()

        if self.Type == RuleType.HEADER:
            self.Predicats_Dependent = [headerBlock[0]]

    def __str__ (self):
        result = ""

        result = result + self.Header.__str__() + " <- "
        for i in range(0, len(self.Body) - 1):
            result = result + self.Body[i].__str__()
            if i + 1 < len(self.Body):
                result += " & "

        return result

    def __repr__ (self):
        return self.__str__()

    def Create_DefinitionZone_HeaderRule (self, idx:int = 0, addon:int = 0) -> list:
        result = []

        result.append(("def DefinitionZone_" + str(idx) + "() -> tuple:", addon))

        if len(self.Header.Matches) is 0:
            result += QueryTree_Create_RuleArgs(self.Header, 0, addon + 1)
        else:
            result += QueryTree_Create_Filter(self.Header, 0, addon + 1)

        result.append(("return start_block_0", addon + 1))
        return result

    def Create_DefinitionZone (self, dictName:str = "MainDict", idx:int = 0, addon:int = 0) -> list:
        """
        create python rule in the "Definition Zone" paradigm
        :param dictName: the name of the dictionary that holds all the data structure
        :return: list of python commands
        """
        result = []

        aboveLst = []

        result.append(("def DefinitionZone_" + str(idx) + "() -> tuple:", addon))
        for i in range(len(self.Body)):
            block = self.Body[i]

            if len(block.Matches) is 0:
                result += QueryTree_Create_RuleArgs(block, i, addon + 1)
            else:
                result += QueryTree_Create_Filter(block, i, addon + 1)

            if block.Type is BlockType.ABOVE:
                aboveLst.append(i)

        result += QueryTree_Create_Join(list(range(0, len(self.Body))), addon = addon + 1)[0]

        if len(aboveLst) > 0:
            result += QueryTree_Create_SelectAbove(aboveLst, self, dictName, addon = addon + 1)
        else:
            result.append(("def_zone = final_join", addon + 1))

        return result

    def Create_CompiledCode (self, total:int, idx:int = 0, addon:int = 0, eps:float = 0.00001) -> list:
        result = []

        result.append(("def Rule_{0}(assigns:np.ndarray, varsPic:np.ndarray) -> tuple:".format(idx), addon))
        result.append(("added, changed = 0,0", addon + 1))

        result.append(("for row in assigns:", addon + 1))

        for i in range(total):
            result.append(("a_{0} = row[varsPic[{0}]]".format(i), addon + 2))

        for block in self.Body:
            if block.Type == BlockType.ANNOTATION:
                tupleKey = ""
                for idx in block.VirtualVarsPic:
                    tupleKey += "a_{0},".format(idx)

                result.append(("{0}=dict_{1}[({2})]".format(block.Notation, block.Predicat, tupleKey), addon + 2))

        block = self.Header
        tupleKey = ""
        for idx in block.VirtualVarsPic:
            tupleKey += "a_{0},".format(idx)

        result.append((
            "if ({0}) not in dict_{1}.keys() and {2} > 0:".format(tupleKey, block.Predicat,
                                                                      block.Notation), addon + 2))
        result.append(("added+=1", addon + 3))
        result.append(("dict_{0}[({1})] = {2}".format(block.Predicat, tupleKey, block.Notation), addon + 3))

        result.append((
            "elif {0} >= dict_{1}[({2})]+({3}):".format(block.Notation, block.Predicat, tupleKey, eps),
            addon + 2))
        result.append(("changed+=1", addon + 3))
        result.append(("dict_{0}[({1})] = {2}".format(block.Predicat, tupleKey, block.Notation), addon + 3))

        result.append(("return added, changed", addon + 1))
        return result

    def Create_CompiledCode_HeaderRule (self, total:int, idx:int = 0, addon:int = 0, eps:float = 0.00001) -> list:
        result = []

        result.append(("def Rule_{0}(assigns:np.ndarray, varsPic:np.ndarray) -> tuple:".format(idx), addon))
        result.append(("changed = 0", addon + 1))

        result.append(("for row in assigns:", addon + 1))

        for i in range(total):
            result.append(("a_{0} = row[varsPic[{0}]]".format(i), addon + 2))

        block = self.Header
        tupleKey = ""
        for idx in block.VirtualVarsPic:
            tupleKey += "a_{0},".format(idx)

        result.append((
            "if {0} >= dict_{1}[({2})]+({3}):".format(block.Notation, block.Predicat, tupleKey, eps), addon + 2))
        result.append(("changed+=1", addon + 3))
        result.append(("dict_{0}[({1})] = {2}".format(block.Predicat, tupleKey, block.Notation), addon + 3))

        result.append(("return 0, changed", addon + 1))
        return result

    def Arrange_Execution (self, idx:int, addon:int = 0):
        if self.Type == RuleType.HEADER:
            self.Code_DefinitionZone = compile(
                _Create_CommandString(self.Create_DefinitionZone_HeaderRule(idx = idx, addon = addon)), "<string>",
                "exec")
            self.Code_Run = compile(_Create_CommandString(
                self.Create_CompiledCode_HeaderRule(len(self.Dictionary), idx = idx, addon = addon)), "<string>",
                                    "exec")
        else:
            self.Code_DefinitionZone = compile(
                _Create_CommandString(self.Create_DefinitionZone(idx = idx, addon = addon)), "<string>", "exec")
            self.Code_Run = compile(
                _Create_CommandString(self.Create_CompiledCode(len(self.Dictionary), idx = idx, addon = addon)),
                "<string>", "exec")

#endregion

#region GAP Compiler
class GAP_Compiler:
    """
    it loads a gap file (of more than 1) and compile the rules to python code
    """

    def __init__ (self, eps:float = 0.00001):
        self.Rules = []
        self.e = eps

    def Load (self, path):
        """
        load a single gap file into the compiler
        :param path: Gap file path
        :return: void
        """
        filer = open(path, "r")

        for line in filer.readlines():
            rule = GAP_Rule(line)
            self.Rules.append(rule)

    def GetPredicats (self) -> list:
        lst = []

        for rule in self.Rules:
            for atom in rule.Predicats:
                if atom not in lst:
                    lst.append(atom)

        return lst

    def Compile (self, dictName:str = "MainDict", addon:int = 0) -> list:
        result = []

        result += QueryTree_Create_Dictionaries(self.GetPredicats(), addon)

        for i in range(len(self.Rules)):
            rule = self.Rules[i]

            if rule.Type == RuleType.HEADER:
                result += rule.Create_DefinitionZone_HeaderRule(idx = i, addon = addon)
                result += rule.Create_CompiledCode_HeaderRule(len(rule.Dictionary), idx = i, addon = addon)
            else:
                result += rule.Create_DefinitionZone(idx = i, addon = addon)
                result += rule.Create_CompiledCode_HeaderRule(len(rule.Dictionary), idx = i, addon = addon)

        return result

#endregion

''''
compiler = GAP_Compiler()
compiler.Load("External/Rules/Pi4m.gap")

rule = compiler.Rules[0]

result = rule.Create_DefinitionZone()

for line in result:
    text, tabsCount = line

    command = ""
    for i in range(tabsCount):
        command += "\t"

    command = command + text

    print(command)
'''''