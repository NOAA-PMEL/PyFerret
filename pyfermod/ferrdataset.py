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
        '''
        (errval, errmsg) = pyferret.run('USE "' + filename + '"')
        if errval != pyferret.FERR_OK:
           raise ValueError(errmsg)
        namesdict = pyferret.getstrdata('..varnames')
        self._filename = filename
        self._varnames = set(namesdict['data'].squeeze().tolist())
        self._ferrvars = { }
        for myname in self._varnames:
           self._ferrvars[myname] = FerrVar(varname=myname, datasetname=self._filename)

    def __repr__(self):
        '''
        Representation to recreate this FerrDataSet.
        Also includes the variable names as variables can be added after creation.
        '''
        infostr = "FerrDataSet('%s') with varnames %s" % (self._filename, str(self._varnames))
        return infostr

    def __getattr__(self, name):
        '''
        Return the FerrVar with the given name.
        '''
        if name not in self._ferrvars:
            raise AttributeError('No attribute or FerrVar with name %s' % name)
        return self._ferrvars[name]

    def __setattr__(self, name, value):
        '''
        If value is a FerrVar, then add this FerrVar to this dataset with the given name.
        If value is not a FerrVar, pass this call onto the parent object.
        '''
        if isinstance(value, FerrVar):
            try:
                value.assignInFerret(name, self._filename)
            except ValueError as ex:
                raise AttributeError('Problems assigning %s in Ferret: %s' % (name, ex.strerror))
            self._varnames.add(name)
            self._ferrvars[name] = value
        else:
            object.__setattr__(self, name, value)

