'''
Represents a data file and the data variables it contains.
'''

import pyferret

class FerDSet(object):
    '''
    A data file and the data variables it contains
    '''

    def __init__(self, filename, qual=None):
        '''
        "Opens" the given NetCDF dataset file in Ferret using the Ferret "USE" command.
        Creates a FerVar for each data variable in this data file and 
        assigns it as an attribute of this class whose name is the variable name.
            filename (string): name of the dataset filename or http address
            qual (string): Ferret qualifiers to be used with the "USE" command
        If both filename is None or empty, an anonymous dataset is returned.
        '''
        if not filename:
            # return an anonymous dataset
            self._filename = ''
            self._dsetname = ''
            self._fervars = { }
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
        self._dsetname = filename[slashidx:]
        if not self._dsetname:
            raise ValueError('invalid dataset name derived from the filename')
        # create a FerVar for each variable in this dataset
        namesdict = pyferret.getstrdata('..varnames')
        self._fervars = { }
        for varname in namesdict['data'].flatten():
            # create a FerVar representing this existing Ferret file variable
            filevar = pyferret.FerVar()
            filevar._markasknownvar(varname, self._dsetname, True)
            # assign this FerVar - uppercase the variable name keys to make case-insensitive
            self._fervars[varname.upper()] = filevar

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
        Representation to recreate this FerDataSet.
        Also includes the variable names as variables can be added after creation.
        '''
        infostr = "FerDSet('%s') using dataset name '%s' and variables %s" % \
                  (self._filename, self._dsetname, str(self.fernames(sort=True)))
        return infostr

    def __eq__(self, other):
        '''
        Two FerDSets are equal if their filenames, datasetnames, and 
        dictionary of FerVar variables are all equal.
        All string values, except for the filename, are compared case-insensitive.
        '''
        if not isinstance(other, FerDSet):
            return NotImplemented
        if self._filename != other._filename:
            return False
        if self._dsetname.upper() != other._dsetname.upper():
            return False
        if self._fervars != other._fervars:
            return False
        return True

    def __ne__(self, other):
        '''
        Two FerDSets are not equal if their filenames, datasetnames, or
        dictionary of FerVar variables are not equal.
        All string values, except for the filename, are compared case-insensitive.
        '''
        if not isinstance(other, FerDSet):
            return NotImplemented
        return not self.__eq__(other)

    def __len__(self):
        '''
        Returns the number of Ferret variables associated with this dataset
        '''
        return len(self._fervars)

    def __getitem__(self, name):
        '''
        Return the Ferret variable (FerVar) with the given name.
        '''
        if not isinstance(name, str):
            raise TypeError('name key is not a string')
        return self._fervars[name.upper()]

    def __setitem__(self, name, value):
        '''
        Creates a copy of value (FerVar), assigns it to Ferret identified by 
        name (string), and adds this copy to this dataset, identified by name.
        '''
        if not isinstance(name, str):
            raise TypeError('name key is not a string')
        if not isinstance(value, pyferret.FerVar):
            raise TypeError('value to be assigned is not a FerVar')
        if self._filename and not self._dsetname:
            raise TypeError('this dataset has been closed')
        # if this name is already assigned to a FerVar, first remove the 
        # Ferret definition that is going to be overwritten; otherwise, 
        # Python's delete of the item in garbage collection will wipe out 
        # the (possibly new) definition as some unknown time.
        try:
            self.__delitem__(name)
        except Exception:
            pass
        # make an anonymous copy of the FerVar (or subclass - the copy 
        # method preserves the class type) and assign it in Ferret.
        newvar = value.copy()
        try:
            newvar._assigninferret(name, self._dsetname)
        except ValueError as ex:
            raise TypeError(str(ex))
        # add this FerVar to this dataset using the uppercase name 
        # to make names case-insenstive 
        self._fervars[name.upper()] = newvar

    def __delitem__(self, name):
        '''
        Removes (cancels) the Ferret variable identified by name (string)
        and removes the FerVar from this dataset.
        '''
        if not isinstance(name, str):
            raise TypeError('name key is not a string')
        uppername = name.upper()
        value = self._fervars[uppername]
        try:
            value._removefromferret()
        except ValueError as ex:
            raise TypeError(str(ex))
        del self._fervars[uppername]

    def __contains__(self, name):
        '''
        Returns whether the Ferret variable name is in this dataset
        '''
        if not isinstance(name, str):
            return False
        return ( name.upper() in self._fervars )

    def __iter__(self):
        '''
        Returns an iterator over the Ferret variable names.
        '''
        return iter(self._fervars)

    def __getattr__(self, name):
        '''
        Returns the Ferret variable (FerVar) with the given name.
        Note that this method is only called when the parent object 
        does not have an attribute with this name.
        '''
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError('no attribute or FerVar with name %s' % name)

    def __setattr__(self, name, value):
        '''
        If value is a FerVar, then creates a copy of this Ferret variable, assigns it 
        to Ferret identified by name (string), and adds it to this dataset identified 
        by name.  If value is not a FerVar, passes this call onto the parent object.
        '''
        if isinstance(value, pyferret.FerVar):
            try:
                self.__setitem__(name, value)
            except TypeError as ex:
                raise AttributeError(str(ex))
        else:
            super(FerDSet, self).__setattr__(name, value)
 
    def __delattr__(self, name):
        '''
        If name is associated with a FerVar, removes (cancels) the Ferret variable 
        identified by name (string) and removes the FerVar from this dataset.
        If name is not associated with FerVar, passes this call onto the parent object.
        '''
        try:
            self.__delitem__(name)
        except TypeError as ex:
            raise AttributeError(str(ex))
        except KeyError:
            try :
                super(FerDSet, self).__delattr__(name)
            except AttributeError:
                raise AttributeError('no attribute or FerVar with name %s' % name)

    def __dir__(self):
        '''
        Returns a list of attributes, include FerVar names, of this object.
        Adds both all-uppercase and all-lowercase FerVar names.
        '''
        mydir = self.fernames(sort=False)
        lcnames = [ name.lower() for name in mydir ]
        mydir.extend( lcnames )
        mydir.extend( dir(super(FerDSet, self)) )
        return mydir

    def fernames(self, sort=False):
        '''
        Returns a list of the names of the current Ferret variables associated with this dataset.
            sort (boolean): sort the list of names?
        '''
        namelist = list( self._fervars.keys() )
        if sort:
            namelist.sort()
        return namelist

    def fervars(self, sort=False):
        '''
        Returns a list of the current Ferret variables associated with this dataset.
            sort (boolean): sort the list of FerVars?
        '''
        varlist = list( self._fervars.values() )
        if sort:
            varlist.sort()
        return varlist

    def close(self):
        '''
        Removes (cancels) all the (non-file) variables in Ferret associated with this dataset,
        then closes (cancels) this dataset in Ferret (which removes the file variables as well).
        Raises a ValueError if there is a problem.
        '''
        # if the dataset is already closed, ignore this command
        if self._filename and not self._dsetname:
            return
        # remove all the Ferret variables associated with this dataset, 
        # ignoring errors from trying to remove file variables.
        for name in self._fervars:
            try:
                # remove this variable from Ferret 
                self._fervars[name]._removefromferret()
            except NotImplementedError:
                pass
        # remove all the FerVar's from _fervars
        self._fervars.clear()
        # nothing else to do if an anonymous dataset
        if not self._dsetname:
            return
        # now remove the dataset
        cmdstr = 'CANCEL DATA %s' % self._dsetname
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('unable to remove dataset %s in Ferret: %s' % self._dsetname)
        # mark this dataset as closed
        self._dsetname = ''

    def show(self, brief=True, qual=''):
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
        if self._filename and not self._dsetname:
            return
        if not isinstance(qual, str):
            raise ValueError('qual (Ferret qualifiers) must be a string')
        if not self._dsetname:
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
            cmdstr += self._dsetname
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('Ferret command "%s" failed: %s' % (cmdstr, errmsg))

