''' Represents a forecast-model-run collection in Ferret '''

import pyferret
from pyferret.ferfmrcvar import _fmrc_var_qualifier

class FerFMRCDSet(pyferret.FerAggDSet):
    '''
    A forecast-model-run collection dataset in Ferret.  Variables in this dataset
    (FerFMRCVar objects) have restrictions on the time and forecast axes, and have
    special transformations to better view the quality of the forecast represented 
    by the variable.
    '''

    def __init__(self, name, dsets, title='', warn=True, hide=False):
        '''
        Creates a forecast-model-run collection dataset in Ferret.  Variables in 
        the given datasets to be aggregated must have a time axes (forecasted time)
        whose values are offset but otherwise match where they overlap (a subset
        of thimes that advances along a single time axis that is the union of all 
        dataset time axes).  The datasets can be given in any order; they will be 
        arranged to have monotonically increasing time subsets.  The aggregated
        variables (FerFMRCVar objects) have special transformations to better view 
        the quality of the forecast represented by this variable.
        '''
        super(FerFMRCDSet, self).__init__(name=name, dsets=dsets, along='F', 
                                          title=title, warn=warn, hide=hide)
        # The datasets have been aggregated along the F axis and all the known 
        # variables added to this dataset's _fervars dictionary.  Turn each of 
        # these FerVar objects into a FerFMRCVar object.
        for uppername in self._fervars.keys():
            if uppername == 'TF_TIMES':
                continue
            self._fervars[uppername] = pyferret.FerFMRCVar(self._fervars[uppername], 
                                                           _fmrc_var_qualifier)

    def __repr__(self):
        '''
        Representation to of this FerFMRCDSet.
        Includes the variable names as variables can be added after creation.
        '''
        infostr = "FerFMRCDSet(name='%s', dsets=%s, hide=%s) with variables %s" % \
                  (self._dsetname, str(self._compdsetnames), 
                   str(self._comphidden), str(self.fernames(sort=True)))
        return infostr

    # Note: Just use the __eq__ and __ne__ methods inherited from FerAggDSet.
    #       No need to make these objects appear to be different from a FerAggDSet.

