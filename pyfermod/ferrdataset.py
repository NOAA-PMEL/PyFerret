'''
Represents a data file and the data variables it contains.

@author: Karl Smith
'''

import pyferret

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
        self._filename = filename
        slashidx = filename.rfind('/') + 1
        self._datasetname = filename[slashidx:]
        namesdict = pyferret.getstrdata('..varnames')
        self._varnames = set( namesdict['data'].squeeze().tolist() )
        self._ferrvars = { }
        for myname in self._varnames:
           # uppercase the variable name keys to make case-insensitive
           self._ferrvars[myname.upper()] = pyferret.FerrVar(varname=myname, datasetname=self._datasetname)

    def __repr__(self):
        '''
        Representation to recreate this FerrDataSet.
        Also includes the variable names as variables can be added after creation.
        '''
        infostr = "FerrDataSet('%s') with varnames %s" % (self._filename, str(list(self._varnames)))
        return infostr

    def __getattr__(self, name):
        '''
        Return the FerrVar with the given name.
        '''
        uppername = name.upper()
        if uppername not in self._ferrvars:
            raise AttributeError('No attribute or FerrVar with name %s' % name)
        return self._ferrvars[uppername]

    def __setattr__(self, name, value):
        '''
        If value is a FerrVar, then add this FerrVar to this dataset with the given name.
        If value is not a FerrVar, pass this call onto the parent object.
        '''
        if isinstance(value, pyferret.FerrVar):
            try:
                value.assignInFerret(name, self._datasetname)
            except ValueError as ex:
                try:
                   # Python3.x
                   msg = ex.strerror
                except AttributeError:
                   # Python2.x
                   msg = ex.message
                raise AttributeError('Problems assigning %s in Ferret: %s' % (name, msg))
            self._varnames.add(name)
            self._ferrvars[name.upper()] = value
        else:
            object.__setattr__(self, name, value)
 
