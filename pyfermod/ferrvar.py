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
            varname (String): name of the Ferret variable
            datasetname (String): name of the dataset containing the variable
            definition (String): Ferret-syntax definition of the variable;
                if varname and datasetname are given and definition is not,
                definition is assigned that of a file variable: <varname>[d=<datasetname>]
        '''
        self._varname = varname
        self._datasetname = datasetname
        if varname and datasetname and not definition:
            self._definition = '%s[d=%s]' % (varname, datasetname)
        else:
            self._definition = definition
        # The _requires list contains FerrVar Ferret names that are know to be used
        # in the definition.  This list is not guarenteed to be complete and is not
        # used in comparisons.
        if varname:
           self._requires = [ varname.upper() ]
        else:
           self._requires = [ ]

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
        return ( self.__cmp__(other) == 0 )

    def __ne__(self, other):
        '''
        Two FerrVars are not equal if any of the following are True:
            they have different Ferret variable name,
            they have different dataset name, or
            they have different defintion.
        All these comparisons are case-insensitive.
        '''
        return ( self.__cmp__(other) != 0 )

    def __lt__(self, other):
        '''
        FerrVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, and finally
        by the definition.
        '''
        return ( self.__cmp__(other) < 0 )

    def __le__(self, other):
        '''
        FerrVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, and finally
        by the definition.
        '''
        return ( self.__cmp__(other) <= 0 )

    def __gt__(self, other):
        '''
        FerrVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, and finally
        by the definition.
        '''
        return ( self.__cmp__(other) > 0 )

    def __ge__(self, other):
        '''
        FerrVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, and finally
        by the definition.
        '''
        return ( self.__cmp__(other) >= 0 )

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
            newdef = '( %s ) + ( %s )' % (self._definition, other._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            newvar._requires.extend(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '( %s ) + %s' % (self._definition, str(other))
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
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
            newdef = '( %s ) + ( %s )' % (other._definition, self._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            newvar._requires.extend(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s + ( %s )' % (str(other), self._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
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
            newdef = '( %s ) - ( %s )' % (self._definition, other._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            newvar._requires.extend(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '( %s ) - %s' % (self._definition, str(other))
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
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
            newdef = '( %s ) - ( %s )' % (other._definition, self._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            newvar._requires.extend(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s - ( %s )' % (str(other), self._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
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
            newdef = '( %s ) * ( %s )' % (self._definition, other._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            newvar._requires.extend(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '( %s ) * %s' % (self._definition, str(other))
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
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
            newdef = '( %s ) * ( %s )' % (other._definition, self._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            newvar._requires.extend(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s * ( %s )' % (str(other), self._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
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
            newdef = '( %s ) / ( %s )' % (self._definition, other._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            newvar._requires.extend(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '( %s ) / %s' % (self._definition, str(other))
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
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
            newdef = '( %s ) / ( %s )' % (other._definition, self._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            newvar._requires.extend(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s / ( %s )' % (str(other), self._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
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
            newdef = '( %s ) ^ ( %s )' % (self._definition, other._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            newvar._requires.extend(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '( %s ) ^ %s' % (self._definition, str(other))
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
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
            newdef = '( %s ) ^ ( %s )' % (other._definition, self._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            newvar._requires.extend(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s ^ ( %s )' % (str(other), self._definition)
            newvar = FerrVar(varname=None,datasetname=None,definition=newdef)
            newvar._requires.extend(self._requires)
            return newvar
        return NotImplemented

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
        if varname.upper() in self._requires:
            raise ValueError('recursive definitions cannot be implemented in Ferret')
        # Assign the variable in Ferret
        cmdstr = 'LET /d=%s %s = %s' % (datasetname, varname, self._definition)
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('Problems defining %s (%s) in Ferret: %s' % (varname, cmdstr, errmsg))
        # Revise the fields in this FerrVar to reflect this assignment
        self._varname = varname
        self._datasetname = datasetname
        self._definition = '%s[d=%s]' % (varname, datasetname)
        self._requires.append(varname.upper())

