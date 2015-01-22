__author__ = "Bar Bokovza"

# region IMPORTS
from collections import defaultdict
import csv

import numpy as np

from Code.opencl import Generate_Empty


# endregion

#region Private Functions
#endregion

#region GAP Data Holder
class GAP_Data:
    """
    The Data Holder
    """

    def __init__ (self):
        """
        Initialization
        """
        self.data = defaultdict()

    def Load (self, path):
        """
        Load a csv file into the data holder
        :param path: csv file path
        :type path: str
        """
        filer = open(path, "r")
        factsReader = csv.DictReader(filer, fieldnames = ["prop", "annotation"], restkey = "args", restval = 0)

        for record in factsReader:
            property = record["prop"]
            if not property in self.data:
                self.data[property] = { }

            property_dict = self.data[property]
            annotation = float(record["annotation"])
            args = tuple(map(int, record["args"]))

            property_dict[args] = annotation

        filer.close()

    def Reset (self):
        """
        Clear all the data
        """
        self.data.clear()

    def GetData (self, name):
        """
        Get all the data of a specified predicat
        :param name: name of the predicat
        :type name:str
        :return: dictionary that holds the data of the predicat
        :rtype: dict
        """
        if not name in self.data.keys():
            return None
        return self.data[name]

    def Generate_NDArray (self, name):
        """
        Create an array from the indexes of the data of predicat
        :param name: Name of predicat
        :type name: str
        :return: Array of Indexes
        :rtype: np.ndarray
        """
        dict = self.GetData(name)
        if dict is None:
            return Generate_Empty(np.int32), Generate_Empty(np.float)

        return np.array(list(dict.keys()), dtype = np.int32)

#endregion


