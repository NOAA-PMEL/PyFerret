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
        namesdict = pyferret.getstrdata('..varnames')
        varnamesarray = namesdict['data'].squeeze()
        self.varnames = [ name for name in varnamesarray ]

    def __repr__(self):
        '''
        Representation to recreate this FerrDataSet
        '''
        repstr = "FerrDataSet('%s')" % self.filename
        return repstr

    def __str__(self):
        '''
        Friendly string representation of this FerrDataSet
        '''
        infostr = "FerrDataSet(filename='%s',varnames=%s)" % (self.filename, str(self.varnames))
        return infostr
