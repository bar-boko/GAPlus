import csv

__author__ = "Bar Bokovza"

# region IMPORTS
import numpy as np
import Code.opencl as pl
# endregion

#region Private Functions
#endregion

#region GAP Data Holder
class GAP_Data:
    def __init__ (self):
        self.data = { }

    def Load (self, path:str):
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
        self.data = { }

    def GetData (self, name) -> dict:
        if not name in self.data.keys():
            return None
        return self.data[name]

    def Generate_NDArray (self, name) -> (np.ndarray, np.ndarray):
        dict = self.GetData(name)
        if dict is None:
            return pl.Generate_Empty(np.int32), pl.Generate_Empty(np.float)

        return np.array(list(dict.keys()), dtype = np.int32)

#endregion


