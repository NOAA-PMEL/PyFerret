'''
Represents a data file and the data variables it contains.

@author: Karl Smith
'''

import pyferret
from ferrvar import FerrVar

class FerrDataSet(object):
    '''
    A data file and the data variables it contains
    '''

    def __init__(self, filename):
        '''
        Opens the given netCDF file in Ferret using the Ferret "USE" command.
        Creates a FerrVar for each data variable in this data file and 
        assigns it as an attribute of this class whose name is the variable name.
        Assigns the list of all variables to the __datavars attribute.
        '''
        self.filename = filename
        (errval, errmsg) = pyferret.run('USE "' + self.filename + '"')
        
