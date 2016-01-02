'''
Represents Ferret variables in Python.

@author: Karl Smith
'''

import numbers
import pyferret

class FerrVar(object):
    '''
    Ferret variable object
    '''

    def __init__(self, varname=None, datasetname=None, definition=None):
        '''
        Creates a Ferret variable without reading or computing any data values.
            varname (string): name of the Ferret variable
            datasetname (string): name of the dataset containing the variable
            definition (string): Ferret-syntax definition of the variable;
                if varname and datasetname are given and definition is not,
                definition is assigned that of a file variable: <varname>[d=<datasetname>]
        '''
        if varname:
            if not isinstance(varname, str):
                raise ValueError("varname is not a string")
        if datasetname:
            if not isinstance(datasetname, str):
                raise ValueError("datasetname is not a string")
        if definition:
            if not isinstance(definition, str):
                raise ValueError("definition is not a string")
        self._varname = varname
        self._datasetname = datasetname
        if varname and datasetname and not definition:
            self._definition = self._ferrname()
        else:
            self._definition = definition
        # The _requires list contains FerrVar Ferret names that are know to be used
        # in the definition.  This list is not guarenteed to be complete and is not
        # used in comparisons.
        self._requires = set()
        if varname:
            self._requires.add(varname.upper())
        # _datagrid is a FerrGrid describing the Ferret grid for the variable.
        # _dataarray and a NumPy ndarray contains the Ferret data for the variable.
        # _dataunit is a string given the unit of the data
        # _missingvalue is the missing value used for the data
        # If not given these will be pulled from Ferret when needed.  
        # If there is a change in the definition of this variable, or any variables 
        # used in its definition, these should be cleared.
        self._datagrid = None
        self._dataarray = None
        self._dataunit = None
        self._missingvalue = None

    def _ferrname(self):
        ''' 
        Returns the Ferret name for this variable, namely 
            <varname>[d=<datasetname>]" 
        Raises ValueError if _varname or _datasetname is not defined
        '''
        if not self._varname:
            raise ValueError('this FerrVar does not contain a Ferret variable name')
        if not self._datasetname:
            raise ValueError('this FerrVar does not contain a dataset name')
        ferrname = '%s[d=%s]' % (self._varname, self._datasetname)
        return ferrname

    def __repr__(self):
        '''
        Representation to recreate this FerrVar
        '''
        infostr = "FerrVar(varname='%s', datasetname='%s', definition='%s')" \
                  % (self._varname, self._datasetname, self._definition)
        return infostr

    def __cmp__(self, other):
        '''
        FerrVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, and finally
        by the definition.  (Used by the "rich comparison" methods.)
        '''
        if not isinstance(other, FerrVar):
            raise NotImplementedError('other is not a FerrVar')
        supper = self._varname.upper()
        oupper = other._varname.upper()
        if supper < oupper:
            return -1
        if supper > oupper:
            return 1
        supper = self._datasetname.upper()
        oupper = other._datasetname.upper()
        if supper < oupper:
            return -1
        if supper > oupper:
            return 1
        supper = self._definition.upper()
        oupper = other._definition.upper()
        if supper < oupper:
            return -1
        if supper > oupper:
            return 1
        return 0

    def __eq__(self, other):
        '''
        Two FerrVars are equal if all of the following are True:
            they have the same Ferret variable name,
            they have the same dataset name, and
            they have the same defintion.
        All these comparisons are case-insensitive.
        '''
        try:
            return ( self.__cmp__(other) == 0 )
        except NotImplementedError:
            return NotImplemented

    def __ne__(self, other):
        '''
        Two FerrVars are not equal if any of the following are True:
            they have different Ferret variable name,
            they have different dataset name, or
            they have different defintion.
        All these comparisons are case-insensitive.
        '''
        try:
            return ( self.__cmp__(other) != 0 )
        except NotImplementedError:
            return NotImplemented

    def __lt__(self, other):
        '''
        FerrVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, and finally
        by the definition.
        '''
        try:
            return ( self.__cmp__(other) < 0 )
        except NotImplementedError:
            return NotImplemented

    def __le__(self, other):
        '''
        FerrVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, and finally
        by the definition.
        '''
        try:
            return ( self.__cmp__(other) <= 0 )
        except NotImplementedError:
            return NotImplemented

    def __gt__(self, other):
        '''
        FerrVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, and finally
        by the definition.
        '''
        try:
            return ( self.__cmp__(other) > 0 )
        except NotImplementedError:
            return NotImplemented

    def __ge__(self, other):
        '''
        FerrVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, and finally
        by the definition.
        '''
        try:
            return ( self.__cmp__(other) >= 0 )
        except NotImplementedError:
            return NotImplemented

    def __nonzero__(self):
        '''
        Returns False if the Ferret variable name, dataset name, and
        definition are all empty or None.  (For Python2.x)
        '''
        if self._varname:
            return True
        if self._datasetname:
            return True
        if self._definition:
            return True
        return False

    def __bool__(self):
        '''
        Returns False if the Ferret variable name, dataset name, and
        definition are all empty or None.  (For Python3.x)
        '''
        return self.__nonzero__()

    def __add__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the sum of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the sum of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerrVar):
            newdef = '(%s) + (%s)' % (self._definition, other._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) + %s' % (self._definition, str(other))
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __radd__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the sum of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the sum of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerrVar):
            newdef = '(%s) + (%s)' % (other._definition, self._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s + (%s)' % (str(other), self._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __sub__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the difference (self - other) of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the difference (self - other) of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerrVar):
            newdef = '(%s) - (%s)' % (self._definition, other._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) - %s' % (self._definition, str(other))
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __rsub__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the difference (other - self) of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the difference (other - self) of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerrVar):
            newdef = '(%s) - (%s)' % (other._definition, self._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s - (%s)' % (str(other), self._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __mul__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the product of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the product of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerrVar):
            newdef = '(%s) * (%s)' % (self._definition, other._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) * %s' % (self._definition, str(other))
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __rmul__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the product of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the product of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerrVar):
            newdef = '(%s) * (%s)' % (other._definition, self._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s * (%s)' % (str(other), self._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __truediv__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the quotient (self / other) of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the quotient (self / other) of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        (For Python3.x)
        '''
        if isinstance(other, FerrVar):
            newdef = '(%s) / (%s)' % (self._definition, other._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) / %s' % (self._definition, str(other))
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __rtruediv__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the quotient (other / self) of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the quotient (other / self) of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        (For Python3.x)
        '''
        if isinstance(other, FerrVar):
            newdef = '(%s) / (%s)' % (other._definition, self._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s / (%s)' % (str(other), self._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __div__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the quotient (self / other) of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the quotient (self / other) of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        (For Python2.x)
        '''
        return self.__truediv__(other)

    def __rdiv__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the quotient (other / self) of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the quotient (other / self) of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        (For Python2.x)
        '''
        return self.__rtruediv__(other)

    def __pow__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the exponentiation (self ^ other) of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the exponentiation (self ^ other) of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerrVar):
            newdef = '(%s) ^ (%s)' % (self._definition, other._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) ^ %s' % (self._definition, str(other))
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __rpow__(self, other):
        '''
        If other is a FerrVar, returns an anonymous FerrVar whose definition 
        is the exponentiation (other ^ self) of the FerrVar definitions.
        If other is Real, returns an anonymous FerrVar whose definition 
        is the exponentiation (other ^ self) of the FerrVar definition with the number.
        If other is not a FerrVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerrVar):
            newdef = '(%s) ^ (%s)' % (other._definition, self._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s ^ (%s)' % (str(other), self._definition)
            newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __neg__(self):
        '''
        Returns an anonymous FerrVar whose definition is 
        the product of -1.0 and this FerrVar definition.
        '''
        newdef = '-1.0 * (%s)' % self._definition
        newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
        newvar._requires.update(self._requires)
        return newvar

    def __pos__(self):
        '''
        Returns an anonymous FerrVar whose definition is 
        the same as this FerrVar definition
        '''
        newvar = FerrVar(varname=None, datasetname=None, definition=self._definition)
        newvar._requires.update(self._requires)
        return newvar

    def __abs__(self):
        '''
        Returns an anonymous FerrVar whose definition is 
        the absolute value of this FerrVar definition.
        '''
        newdef = 'abs(%s)' % self._definition
        newvar = FerrVar(varname=None, datasetname=None, definition=newdef)
        newvar._requires.update(self._requires)
        return newvar

    def _defineinferret(self, varname, datasetname):
        '''
        Defines this FerrVar in Ferret using the given variable name 
        associated with the given dataset name.
            varname (string): name for the variable in Ferret
            datasetname (string): name of the dataset to contain the variable
        Raises a ValueError if there is a problem.
        '''
        if not self._definition:
            raise ValueError('this FerrVar does not contain a definition')
        if not varname:
            raise ValueError('variable name to be assigned is not given')
        if not datasetname:
            raise ValueError('name of dataset to contain the variable is not given')
        if varname.upper() in self._requires:
            raise ValueError('recursive definitions cannot be implemented in Ferret')
        # Assign the variable in Ferret
        cmdstr = 'DEFINE VAR /d=%s %s = %s' % (datasetname, varname, self._definition)
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('Problems defining %s (%s) in Ferret: %s' % (varname, cmdstr, errmsg))
        # Revise the fields in this FerrVar to reflect this assignment
        self._varname = varname
        self._datasetname = datasetname
        self._definition = self._ferrname()
        self._requires.add(varname.upper())

    def _cancelinferret(self):
        '''
        Cancels (deletes) this variable in Ferret.
        Raises a ValueError if there is a problem.
        '''
        cmdstr = 'CANCEL VAR %s' % self._ferrname()
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('Problems cancelling %s (%s) in Ferret: %s' % (self._varname, cmdstr, errmsg))

    def varname(self):
        ''' Returns the Ferret name of this Ferret variable '''
        return self._varname

    def datasetname(self):
        ''' Returns the dataset name of this Ferret variable '''
        return self._datasetname

    def definition(self):
        ''' Returns the Ferret definition of this Ferret variable '''
        return self._definition

    def clean(self):
        '''
        Clears the grid and data stored in this FerrVar.  After this call, any 
        request for the grid or data will automatically fetch the latest values 
        from Ferret.  This method should be called anytime there is a change 
        in the definition of this variable, or a variable this variable uses.
        '''
        self._datagrid = None
        self._dataarray = None
        self._dataunit = None
        self._missingvalue = None

    def fetch(self):
        '''
        Retrieves the grid and data for this Ferret variable from Ferret.
        This method is automatically called before returning the grid or data 
        for the first time for this variable.  This can be called to update
        the grid or data in this FerrVar after any change in the definition 
        of the variable.  Alternatively, cleardata can be called to clear any
        stored grid and data, delaying the update from Ferret until the grid
        or data is requested.
        Raises a ValueEror if problems occur.
        '''
        ferrname = self._ferrname()
        datadict = pyferret.getdata(ferrname, False)
        self._datagrid = pyferret.FerrGrid(gridname=ferrname,
                                           axistypes=datadict["axis_types"], 
                                           axiscoords=datadict["axis_coords"], 
                                           axisunits=datadict["axis_units"], 
                                           axisnames=datadict["axis_names"])
        self._dataarray = datadict["data"]
        self._dataunit = datadict["data_unit"]
        self._missingvalue = datadict["missing_value"]

