import csv

__author__ = "Bar Bokovza"

# region IMPORTS
import numpy as np
# endregion

def get_empty_array ( dtype ) -> np.ndarray:
    return np.zeros( 0, dtype = dtype )

class GAP_Data:
    data = {}

    def __init__ ( self ):
        self.data = {}

    def load ( self, path:str ):
        filer = open( path, "r" )
        factsReader = csv.DictReader( filer, fieldnames = ["prop", "annotation"], restkey = "args", restval = 0 )
        size = 0

        for record in factsReader:
            property = record ["prop"]
            if (not property in self.data):
                self.data [property] = {}

            property_dict = self.data [property]
            annotation = float( record ["annotation"] )
            args = tuple( map( int, record ["args"] ) )

            property_dict [args] = annotation

        filer.close( )

    def reset ( self ):
        self.data = {}

    def get_data ( self, name ) -> dict:
        if not name in self.data.keys( ):
            return None
        return self.data [name]

    def to_NDArray ( self, name ) -> (np.ndarray, np.ndarray):
        dict = self.get_data( name )
        if dict is None:
            return get_empty_array( np.int32 )

        return np.array( list( dict.keys( ) ), dtype = np.int32 )




