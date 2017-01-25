''' Represents a forecast-model-run collection in Ferret '''

from __future__ import print_function

import pyferret

class FerFMRC(pyferret.FerAggDSet):
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
        super(FerFMRCVar, self).__init__(name=name, dsets=dsets, along='F', 
                                         title=title, warn=warn, hide=hide)

