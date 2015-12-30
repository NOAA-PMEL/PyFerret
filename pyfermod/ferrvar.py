import numpy

'''
Represents Ferret variables in Python.

@author: Karl Smith
'''

class FerrVar(object):
    '''
    Ferret variable object
    '''

    def __init__(self, varname=None, datasetname=None, definition=None):
        '''
        Creates a Ferret variable without reading or computing any data values.
            varname (String): name of the Ferret variable
            datasetname (String): name of the dataset containing the variable
            definition (String): Ferret-syntax definition of the variable;
                if varname and datasetname are given and definition is not,
                definition is assigned that of a file variable: <varname>[d=<datasetname>]
        '''
        self._varname = varname
        self._datasetname = datasetname
        if varname and datasetname and not definition:
           self._definition = '%s[d="%s"]' % (varname, datasetname)
        else:
           self._definition = definition

    def __repr__(self):
        '''
        Representation to recreate this FerrVar
        '''
        infostr = "FerrVar(varname='%s', datasetname='%s', definition='%s')" \
                  % (self._varname, self._datasetname, self._definition)
        return infostr

    def assignInFerret(self, varname, datasetname):
        '''
        Assigns this FerrVar in Ferret using the given variable name 
        associated with the given dataset name.
            varname (String): name for the variable in Ferret
            datasetname (String): name of the dataset to contain the variable
        '''
        if not self._definition:
            raise ValueError('this FerrVar does not contain a definition')
        if not varname:
            raise ValueError('variable name to be assigned is not given')
        if not datasetname:
            raise ValueError('name of dataset to contain the variable is not given')
        # Assign the variable in Ferret
        cmdstr = 'LET /d="%s" %s = %s' % (datasetname, varname, self._definition)
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('Problems defining %s (%s) in Ferret: %s' % (name, cmdstr, errmsg))
        # Revise the fields in this FerrVar to reflect this assignment
        self._varname = varname
        self._datasetname = datasetname
        self._definition = '%s[d="%s"]' % (varname, datasetname)

