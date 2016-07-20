'''
Represents an aggregation of data sets
'''

import pyferret
from pyferret.ferdset import _anonymous_dataset_qualifier

class FerAggDSet(pyferret.FerDSet):
    '''
    An aggregation of data sets and the variables they have in common
    '''

    def __init__(self, name, dsets, along='T', title='', warn=True, hide=False):
        '''
        Aggregates the given list of datasets along the given axis using the 
        Ferret "DEFINE DATA /AGGREGATE" command.  Creates a FerVar for each data 
        variable in common among these datasets, and assigns it as an attribute 
        of this class instance using the variable name.  
            name (string): Ferret name for this aggregated dataset
            dsets (sequence of strings and/or FerDSets): datasets to aggregate.
                A string will be interpreted as a filename for creating a FerDSet.
            along ('T', 'E', 'F'): axis along which to aggregate the datasets
            title (string): title for the dataset for plots and listing;
                if not given, the Ferret name for the dataset will be used
            warn (bool): issue warning messages about variables not in common among
                all member datasets (either not present or not using the same grid)
            hide (bool): hide the member datasets in standard Ferret listings 
                such as with pyferret.showdata()
        '''
        # Create an empty dataset with the given Ferret name
        super(FerAggDSet, self).__init__('', qual=_anonymous_dataset_qualifier)
        if not isinstance(name, str):
            raise ValueError('Ferret name for the aggregate dataset must be astring')
        aggname = name.strip()
        if not aggname:
            raise ValueError('Ferret name for the aggregate dataset is blank')
        self._filename = aggname
        self._dsetname = aggname
        # Need to keep the given order of component datasets
        self._compdsetnames = [ ]
        # But still use a dictionary with uppercase names for keys
        self._compdsets = { }
        # Create the DEFINE DATA /AGGREGATE Ferret command, creating
        # and saving component FerDSets as needed
        if along not in ('T', 'E', 'F'):
            raise ValueError("along must be one of 'T', 'E', or 'F'")
        cmdstr = 'DEFINE DATA/AGGREGATE/' + along
        if title:
            cmdstr += '/TITLE="' + str(title) + '"'
        if not warn:
            cmdstr += '/QUIET'
        if hide:
            cmdstr += '/HIDE'
        cmdstr += ' ' + aggname + ' = '
        firstone = True
        if not ( isinstance(dsets, tuple) or isinstance(dsets, list) ):
            raise ValueError('dsets must be a tuple or list of strings and/or FerDSets')
        for myitem in dsets:
            if isinstance(myitem, str):
                mydset = pyferret.FerDSet(myitem)
            elif isinstance(myitem, pyferret.FerDSet):
                mydset = myitem
            else:
                raise ValueError('dsets must be a tuple or list of strings and/or FerDSets')
            if mydset._dsetname.upper() in self._compdsets:
                raise ValueError('duplicate dataset name ' + mydset._dsetname)
            if not firstone:
                cmdstr += ', '
            else:
                firstone = False
            cmdstr += mydset._dsetname
            self._compdsetnames.append(mydset._dsetname)
            self._compdsets[mydset._dsetname.upper()] = mydset
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError(errmsg)

        # create a FerVar for each variable in this dataset
        namesdict = pyferret.getstrdata('..varnames')
        for varname in namesdict['data'].flatten():
            # create a FerVar representing this existing Ferret aggregated file variable
            filevar = pyferret.FerVar()
            filevar._markasknownvar(varname, self._dsetname, True)
            # assign this FerVar - uppercase the variable name keys to make case-insensitive
            self._fervars[varname.upper()] = filevar
            # keep a original-case version of the name
            self._fervarnames.add(varname)


    def __repr__(self):
        '''
        Representation to of this FerAggDSet.
        Includes the variable names as variables can be added after creation.
        '''
        infostr = "FerAggDSet(name='%s', dsets=%s) with variables %s" % \
                  (self._dsetname, str(self._compdsetnames), str(self.fernames(sort=True)))
        return infostr


    def __eq__(self, other):
        '''
        Two FerAddDSets are equal if their Ferret names, lists of aggregated 
        dataset names, and dictionary of FerVar variables are all equal.
        All string values are compared case-insensitive.
        '''
        if not isinstance(other, pyferret.FerDSet):
            return NotImplemented
        if not isinstance(other, FerAggDSet):
            return False
        if not super(FerAggDSet, self).__eq__(other):
            return False
        if len(self._compdsetnames) != len(other._compdsetnames):
            return False
        for k in xrange(len(self._compdsetnames)):
            if self._compdsetnames[k].upper() != other._compdsetnames[k].upper():
                return False
        return True


    def __ne__(self, other):
        '''
        Two FerDSets are not equal if their Ferret names, lists of aggregated
        dataset names, or dictionary of FerVar variables are not equal.
        All string values are compared case-insensitive.
        '''
        if not isinstance(other, pyferret.FerDSet):
            return NotImplemented
        return not self.__eq__(other)


    def getdsetnames(self):
        '''
        Returns a copy of the list of component dataset names (original-case)
        in the order of their aggregation.
        '''
        return list(self._compdsetnames)


    def getdsets(self):
        '''
        Returns a list of component FerDSet datasets 
        in the order of their aggregation.
        '''
        return [ self._compdsets[name.upper()] for name in self._compdsetnames ]


    def close(self):
        '''
        Removes (cancels) all the variables in Ferret associated with this dataset,
        then closes (cancels) this dataset in Ferret.  If the aggregated dataset was 
        created with hide=True, this will close (cancel) all the component datasets 
        as well.  Raises a ValueError if there is a problem.
        '''
        # if the dataset is already closed, ignore this command
        if not self._dsetname:
            return
        # run the Ferret CANCEL commands in FerDSet.close
        super(FerAggDSet, self).close()
        # clear the list of dataset names and dictionary of datasets
        self._compdsets.clear()
        self._compdsetnames = [ ]

