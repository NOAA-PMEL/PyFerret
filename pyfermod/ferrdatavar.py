'''
Represents Ferret data variables whose data comes directly from an associated data file.

@author: Karl Smith
'''
from ferrvar import FerrVar

class FerrDataVar(FerrVar):
    '''
    A Ferret data variable whose data comes directly from a given data file. 
    '''


    def __init__(self, name, dataset):
        '''
        Represents a Ferret data variable with the given name 
        associated with the given dataset.  No data is read 
        for this data variable.
        '''
        self.name = name
        self.dataset = dataset
        self.definition = self.name + "[dset=" + self.dataset + "]"
