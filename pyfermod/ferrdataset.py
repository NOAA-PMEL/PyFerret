'''
Represents a data file and the data variables it contains.

@author: Karl Smith
'''

import pyferret

class FerrDataSet(object):
    '''
    A data file and the data variables it contains
    '''

    def __init__(self, filename, qual=None):
        '''
        "Opens" the given NetCDF dataset file in Ferret using the Ferret "USE" command.
        Creates a FerrVar for each data variable in this data file and 
        assigns it as an attribute of this class whose name is the variable name.
            filename (string): name of the dataset filename or http address
            qual (string): Ferret qualifiers to be used with the "USE" command
        '''
        if not filename:
            raise ValueError('no filename given')
        if qual:
            cmdstr = 'USE %s "%s"' % (qual, filename)
        else:
            cmdstr = 'USE "%s"' % filename
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError(errmsg)
        # record the filename
        self._filename = filename
        # record the name of the dataset in Ferret
        slashidx = filename.rfind('/') + 1
        self._datasetname = filename[slashidx:]
        if not self._datasetname:
            raise ValueError('invalid dataset name derived from the filename')
        # create a FerrVar for each variable in this dataset
        namesdict = pyferret.getstrdata('..varnames')
        self._ferrvars = { }
        for name in namesdict['data'].flatten():
            # uppercase the variable name keys to make case-insensitive
            self._ferrvars[name.upper()] = pyferret.FerrVar(varname=name, 
                           datasetname=self._datasetname, isfilevar=True)

    def __del__(self):
        '''
        Removes this dataset from Ferret.
        '''
        try:
            self.close()
            # ignore any errors (Ferret or any other)
        except Exception:
            pass

    def __repr__(self):
        '''
        Representation to recreate this FerrDataSet.
        Also includes the variable names as variables can be added after creation.
        '''
        infostr = "FerrDataSet('%s') with Ferret name %s and variables %s" % \
                  (self._filename, self._datasetname, str(self.varnames()))
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
        if not self._datasetname:
            raise TypeError('this dataset has been removed from Ferret')
        try:
            value.assign(name, self._datasetname)
        except ValueError as ex:
            raise TypeError('unable to assign variable %s in Ferret: %s' % (name, str(ex)))
        self._ferrvars[name.upper()] = value

    def __delitem__(self, name):
        '''
        Removes (cancels) the Ferret variable identified by name (string)
        and removes the FerrVar from this dataset.
        '''
        if not isinstance(name, str):
            raise TypeError('name key is not a string')
        uppername = name.upper()
        value = self._ferrvars[uppername]
        try:
            value.remove()
        except ValueError as ex:
            raise TypeError('unable to remove variable %s in Ferret: %s' % (name, str(ex)))
        del self._ferrvars[uppername]

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
        Note that this method is only called when the parent object 
        does not have an atrribute with this name.
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
            super(FerrDataSet, self).__setattr__(name, value)
 
    def __delattr__(self, name):
        '''
        If name is associated with a FerrVar, removes (cancels) the Ferret variable 
        identified by name (string) and removes the FerrVar from this dataset.
        If name is not associated with FerrVar, passes this call onto the parent object.
        '''
        try:
            self.__delitem__(name)
        except TypeError as ex:
            raise AttributeError(str(ex))
        except KeyError:
            try :
                super(FerrDataSet, self).__delattr__(name)
            except AttributeError:
                raise AttributeError('no attribute or FerrVar with name %s' % name)

    def __dir__(self):
        '''
        Returns a list of attributes, include FerrVar names, of this object.
        '''
        mydir = self.varnames(False)
        mydir.extend( dir(super(FerrDataSet, self)) )
        return mydir

    def varnames(self, sort=True):
        '''
        Returns a list of the names of the current Ferret variables associated with this dataset
            sort (boolean): sort the list of names?
        '''
        namelist = list( self._ferrvars.keys() )
        if sort:
            namelist.sort()
        return namelist

    def close(self):
        '''
        Removes (cancels) all the (non-file) variables in Ferret associated with this dataset,
        then closes (cancels) this dataset in Ferret (which removes the file variables as well).
        '''
        # ignore the call if already removed
        if not self._datasetname:
            return
        # remove all the Ferret variables associated with this dataset, 
        # ignoring errors (file variables will fail)
        for name in self._ferrvars:
            try:
                # remove this variable from Ferret 
                self._ferrvars[name].remove()
            except Exception:
                pass
        # remove all the FerrVar's from _ferrvars
        self._ferrvars.clear()
        # now remove the dataset
        cmdstr = 'CANCEL DATA %s' % self._datasetname
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('cannot remove dataset %s in Ferret: %s' % self._datasetname)
        # mark this dataset as invalid
        self._datasetname = ''

    def showdataset(self, brief=True, qual=''):
        '''
        Show the Ferret information about this dataset.  This uses the Ferret
        SHOW DATA command to create and display the information.
            brief (boolean): if True (default), a brief report is shown;
                otherwise a full report is shown.
            qual (string): Ferret qualifiers to add to the SHOW DATA command
        '''
        if not isinstance(qual, str):
            raise ValueError('qual (Ferret qualifiers) must be a string')
        cmdstr = 'SHOW DATA'
        if not brief:
            cmdstr += '/FULL'
        if qual:
            cmdstr += qual
        cmdstr += ' '
        cmdstr += self._datasetname
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('Ferret command "%s" failed: %s' % (cmdstr, errmsg))

