import numpy

'''
Represents Ferret variables in Python.
@author: Karl Smith
'''

class FerrVar(object):
    '''
    Ferret variable object
    '''


    def __init__(self):
        '''
        Creates a Ferret variable without reading or computing any data values.
        '''
        self.name = None
        self.dataset = None
        self.definition = None
        self.data = None

