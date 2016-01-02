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
        self._ferrvars = { }
        for myname in namesdict['data'].squeeze().tolist():
           # uppercase the variable name keys to make case-insensitive
           self._ferrvars[myname.upper()] = pyferret.FerrVar(varname=myname, datasetname=self._datasetname)

    def __repr__(self):
        '''
        Representation to recreate this FerrDataSet.
        Also includes the variable names as variables can be added after creation.
        '''
        infostr = "FerrDataSet('%s') with varnames %s" % (self._filename, str(self.varnames()))
        return infostr

    def __eq__(self, other):
        '''
        Returns if this FerrDataSet is equal to the other FerrDataSet.
        All strings are compared case-insensitive.
        '''
        if not isinstance(other, FerrDataSet):
            return NotImplemented
        if self._filename.upper() != other._filename.upper():
            return False
        if self._datasetname.upper() != other._datasetname.upper():
            return False
        if self._ferrvars != other._ferrvars:
            return False
        return True

    def __ne__(self, other):
        '''
        Returns if this FerrDataSet is not equal to the other FerrDataSet.
        All strings are compared case-insensitive.
        '''
        if not isinstance(other, FerrDataSet):
            return NotImplemented
        if self._filename.upper() != other._filename.upper():
            return True
        if self._datasetname.upper() != other._datasetname.upper():
            return True
        if self._ferrvars != other._ferrvars:
            return True
        return False

    def __len__(self):
        '''
        Returns the number of Ferret variables associated with this dataset
        '''
        return len(self._ferrvars)

    def __getitem__(self, name):
        '''
        Return the Ferret variable (FerrVar) with the given name.
        '''
        if not isinstance(name, str):
            raise TypeError('name key is not a string')
        return self._ferrvars[name.upper()]

    def __setitem__(self, name, value):
        '''
        Assigns the value (FerrVar) to Ferret identified by name (string),
        and adds value to this dataset identified by name.
        '''
        if not isinstance(name, str):
            raise TypeError('name key is not a string')
        if not isinstance(value, pyferret.FerrVar):
            raise TypeError('value to be assigned is not a FerrVar')
        try:
            value._defineinferret(name, self._datasetname)
        except ValueError as ex:
            raise TypeError('unable to assign variable %s in Ferret: %s' % (name, str(ex)))
        self._ferrvars[name.upper()] = value

    def __delitem__(self, name):
        '''
        Cancels (deletes) the Ferret variable identified by name (string)
        and removes the FerrVar from this dataset.
        '''
        if not isinstance(name, str):
            raise TypeError('name key is not a string')
        value = self._ferrvars[name.upper()]
        try:
            value._cancelinferret()
        except ValueError as ex:
            raise TypeError('unable to cancel variable %s in Ferret: %s' % (name, str(ex)))
        del self._ferrvars[name.upper()]

    def __contains__(self, name):
        '''
        Returns whether the Ferret variable name is in this dataset
        '''
        if not isinstance(name, str):
            return False
        return ( name.upper() in self._ferrvars )

    def __iter__(self):
        '''
        Returns an iterator over the Ferret variable names.
        '''
        return iter(self._ferrvars)

    def __getattr__(self, name):
        '''
        Returns the Ferret variable (FerrVar) with the given name.
        '''
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError('no attribute or FerrVar with name %s' % name)

    def __setattr__(self, name, value):
        '''
        If value is a FerrVar, then assigns the Ferret variable value (FerrVar) 
        to Ferret identified by name (string), and adds value to this dataset identified by name.
        If value is not a FerrVar, passes this call onto the parent object.
        '''
        if isinstance(value, pyferret.FerrVar):
            try:
                self.__setitem__(name, value)
            except TypeError as ex:
                raise AttributeError(str(ex))
        else:
            object.__setattr__(self, name, value)
 
    def __delattr__(self, name):
        '''
        If name is associated with a FerrVar, cancels (deletes) the Ferret variable 
        identified by name (string) and removes the FerrVar from this dataset.
        If name is not associated with FerrVar, passes this call onto the parent object.
        '''
        try:
            self.__delitem__(name)
        except TypeError as ex:
            raise AttributeError(str(ex))
        except KeyError:
            object.__delattr__(self, name, value)

    def varnames(self):
        '''
        Returns a list of the names of the current Ferret variables associated with this dataset
        '''
        namelist = list(self._ferrvars.keys())
        namelist.sort()
        return namelist

