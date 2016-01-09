'''
Represents a data file and the data variables it contains.
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
        If both filename is None or empty, an anonymous dataset is returned.
        '''
        if not filename:
            # return an anonymous dataset
            self._filename = ''
            self._datasetname = ''
            self._ferrvars = { }
            return
        # tell Ferret to use this dataset
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
        infostr = "FerrDataSet('%s') using dataset name '%s' and variables %s" % \
                  (self._filename, self._datasetname, str(self.varnames()))
        return infostr

    def __eq__(self, other):
        '''
        Two FerrDatSets are equal if their filenames, datasetnames, and 
        dictionary of FerrVar variables are all equal.
        All string values, except for the filename, are compared case-insensitive.
        '''
        if not isinstance(other, FerrDataSet):
            return NotImplemented
        if self._filename != other._filename:
            return False
        if self._datasetname.upper() != other._datasetname.upper():
            return False
        if self._ferrvars != other._ferrvars:
            return False
        return True

    def __ne__(self, other):
        '''
        Two FerrDatSets are not equal if their filenames, datasetnames, or
        dictionary of FerrVar variables are not equal.
        All string values, except for the filename, are compared case-insensitive.
        '''
        if not isinstance(other, FerrDataSet):
            return NotImplemented
        return not self.__eq__(other)

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
        Creates a copy of value (FerrVar), assigns it to Ferret identified by 
        name (string), and adds this copy to this dataset, identified by name.
        '''
        if not isinstance(name, str):
            raise TypeError('name key is not a string')
        if not isinstance(value, pyferret.FerrVar):
            raise TypeError('value to be assigned is not a FerrVar')
        if self._filename and not self._datasetname:
            raise TypeError('this dataset has been closed')
        # if this name is already assigned to a FerrVar, first remove the 
        # Ferret definition that is going to be overwritten; otherwise, 
        # Python's delete of the item in garbage collection will wipe out 
        # the (possibly new) definition as some unknown time.
        try:
            self.__delitem__(name)
        except Exception:
            pass
        # make an anonymous copy of the FerrVar (or subclass) by calling its copy method 
        newvar = value.copy()
        try:
            newvar._assigninferret(name, self._datasetname)
        except ValueError as ex:
            raise TypeError(str(ex))
        self._ferrvars[name.upper()] = newvar

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
            value._removefromferret()
        except ValueError as ex:
            raise TypeError(str(ex))
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
        If value is a FerrVar, then creates a copy of this Ferret variable, assigns it 
        to Ferret identified by name (string), and adds it to this dataset identified 
        by name.  If value is not a FerrVar, passes this call onto the parent object.
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
        Raises a ValueError if there is a problem.
        '''
        # if the dataset is already closed, ignore this command
        if self._filename and not self._datasetname:
            return
        # remove all the Ferret variables associated with this dataset, 
        # ignoring errors from trying to remove file variables.
        for name in self._ferrvars:
            try:
                # remove this variable from Ferret 
                self._ferrvars[name]._removefromferret()
            except NotImplementedError:
                pass
        # remove all the FerrVar's from _ferrvars
        self._ferrvars.clear()
        # nothing else to do if an anonymous dataset
        if not self._datasetname:
            return
        # now remove the dataset
        cmdstr = 'CANCEL DATA %s' % self._datasetname
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('unable to remove dataset %s in Ferret: %s' % self._datasetname)
        # mark this dataset as closed
        self._datasetname = ''

    def showdataset(self, brief=True, qual=''):
        '''
        Show the Ferret information about this dataset.  This uses the Ferret
        SHOW DATA command to create and display the information.
            brief (boolean): if True (default), a brief report is shown;
                otherwise a full report is shown.
            qual (string): Ferret qualifiers to add to the SHOW DATA command
        If this is an anonymous dataset (no dataset name), the Ferret 
        SHOW VAR/USER command is used instead to show all variables
        created by anonymous datasets.
        '''
        # if the dataset is closed, ignore this command
        if self._filename and not self._datasetname:
            return
        if not isinstance(qual, str):
            raise ValueError('qual (Ferret qualifiers) must be a string')
        if not self._datasetname:
            cmdstr = 'SHOW VAR/USER'
            if qual:
                cmdstr += qual
        else:
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

