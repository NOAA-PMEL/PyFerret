'''
Represents Ferret variables in Python.

@author: Karl Smith
'''

import numbers
import pyferret

# common regridding methods
REGRID_LINEAR = "@LIN"
REGRID_AVERAGE = "@AVE"
REGRID_ASSOCIATE = "@ASN"
REGRID_MEAN = "@BIN"
REGRID_NEAREST = "@NRST"
REGRID_MIN = "@MIN"
REGRID_MAX = "@MAX"
REGRID_EXACT = "@XACT"

class FerrVar(object):
    '''
    Ferret variable object
    '''

    def __init__(self, varname='', datasetname='', definition='', isfilevar=False):
        '''
        Creates a Ferret variable without reading or computing any data values.
            varname (string): name of the Ferret variable
            datasetname (string): name of the dataset containing the variable
            definition (string): Ferret-syntax definition of the variable;
                if varname is given and definition is not,
                definition is assigned that of a known variable: 
                    <varname>[d=<datasetname>]   if datasetname is given, or
                    <varname>                    if datasetname is not given
        '''
        # Record the Ferret variable name, or an empty string if not given
        if varname:
            if not isinstance(varname, str):
                raise ValueError("varname is not a string")
            self._varname = varname
        else:
            self._varname = ''
        # Record the dataset name, or an empty string if not given
        if datasetname:
            if not isinstance(datasetname, str):
                raise ValueError("datasetname is not a string")
            self._datasetname = datasetname
        else:
            self._datasetname = ''
        # Record or generate the definition, or set to an empty string
        if definition:
            if not isinstance(definition, str):
                raise ValueError("definition is not a string")
            self._definition = definition
        elif varname:
            self._definition = self.ferretname()
        else:
            self._definition = ''
        # Record whether this is a file variable (and thus should not be cancelled)
        if isfilevar:
            self._isfilevar = True
        else:
            self._isfilevar = False
        # The _requires list contains FerrVar Ferret names that are know to be used
        # in the definition.  This list is not guarenteed to be complete and is not
        # used in comparisons.
        self._requires = set()
        if varname:
            self._requires.add(varname.upper())
        # Call the clean method to create and set the defaults for 
        # _datagrid, _dataarray, _dataunit, and _missingvalue.
        #     _datagrid is a FerrGrid describing the Ferret grid for the variable.
        #     _dataarray and a NumPy ndarray contains the Ferret data for the variable.
        #     _dataunit is a string given the unit of the data
        #     _missingvalue is the missing value used for the data
        self.clean()

    def __del__(self):
        '''
        Calls remove to remove this variable, if possible, from Ferret.
        Any error are ignored.
        '''
        # Ignore for obvious fail cases
        if self._isfilevar or not self._varname:
            return
        # Try to remove from Ferret but ignore errors
        try:
            self.remove()
        except Exception:
            pass

    def ferretname(self):
        ''' 
        Returns the Ferret name for this variable, namely
            <_varname>[d=<_datasetname>]
        if _datasetname is given; otherwise just
            <_varname>
        Raises ValueError if _varname is not defined
        '''
        if not self._varname:
            raise ValueError('this FerrVar does not contain a Ferret variable name')
        if self._datasetname:
            ferrname = '%s[d=%s]' % (self._varname, self._datasetname)
        else:
            ferrname = '%s' % self._varname
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
        definition are all empty.  (For Python2.x)
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
        definition are all empty.  (For Python3.x)
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
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) + %s' % (self._definition, str(other))
            newvar = FerrVar(definition=newdef)
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
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s + (%s)' % (str(other), self._definition)
            newvar = FerrVar(definition=newdef)
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
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) - %s' % (self._definition, str(other))
            newvar = FerrVar(definition=newdef)
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
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s - (%s)' % (str(other), self._definition)
            newvar = FerrVar(definition=newdef)
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
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) * %s' % (self._definition, str(other))
            newvar = FerrVar(definition=newdef)
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
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s * (%s)' % (str(other), self._definition)
            newvar = FerrVar(definition=newdef)
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
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) / %s' % (self._definition, str(other))
            newvar = FerrVar(definition=newdef)
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
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s / (%s)' % (str(other), self._definition)
            newvar = FerrVar(definition=newdef)
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
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) ^ %s' % (self._definition, str(other))
            newvar = FerrVar(definition=newdef)
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
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s ^ (%s)' % (str(other), self._definition)
            newvar = FerrVar(definition=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented

    def __neg__(self):
        '''
        Returns an anonymous FerrVar whose definition is 
        the product of -1.0 and this FerrVar definition.
        '''
        newdef = '-1.0 * (%s)' % self._definition
        newvar = FerrVar(definition=newdef)
        newvar._requires.update(self._requires)
        return newvar

    def __pos__(self):
        '''
        Returns an anonymous FerrVar whose definition is 
        the same as this FerrVar definition.
        '''
        newvar = FerrVar(definition=self._definition)
        newvar._requires.update(self._requires)
        return newvar

    def __abs__(self):
        '''
        Returns an anonymous FerrVar whose definition is 
        the absolute value of this FerrVar definition.
        '''
        newdef = 'abs(%s)' % self._definition
        newvar = FerrVar(definition=newdef)
        newvar._requires.update(self._requires)
        return newvar

    def __getitem__(self, key):
        '''
        Returns an anonymous FerrVar whose definition is a subset 
        of this FerrVar.  This FerrVar must be assigned in Ferret.
            key is an int, float, string, int slice, float slice, 
                string slice, or a tuple of these values.
                 - int are interpreted as index/indices
                 - floats are interpreted as axis values
                 - strings are interpreted as axis values possibly with units
        Units in a string designate an axis; otherwise the index
        within the given tuple (or zero if not a tuple) specifies the axis.
        For example ['20N':'50N'] will always be a latitude subset.
        TODO: handle step values
        '''
        if not self._varname:
            raise NotImplementedError('slicing can only be performed on variables assigned in Ferret')
        if key == None:
            raise KeyError('None is not a valid key')
        coordlimits = [ None ] * pyferret.MAX_FERRET_NDIM
        indexlimits = [ None ] * pyferret.MAX_FERRET_NDIM
        changed = False
        # TODO: clean-up; lots of repeated code
        if isinstance(key, tuple):
            for k in xrange(len(key)):
                piece = key[k]
                if piece == None:
                    continue
                if isinstance(piece, slice):
                    try:
                        (axtype, start, stop, step) = pyferret.FerrGrid._parsegeoslice(piece)
                    except Exception as ex:
                        raise KeyError('%s is not valid: %s' % (str(piece), str(ex)))
                    if axtype == pyferret.AXISTYPE_LONGITUDE:
                        if coordlimits[pyferret.X_AXIS] or indexlimits[pyferret.X_AXIS]:
                            raise KeyError('two longitude slices given')
                        coordlimits[pyferret.X_AXIS] = '%s:%s' % (str(start), str(stop))
                        changed = True
                    elif axtype == pyferret.AXISTYPE_LATITUDE:
                        if coordlimits[pyferret.Y_AXIS] or indexlimits[pyferret.Y_AXIS]:
                            raise KeyError('two latitude slices given')
                        coordlimits[pyferret.Y_AXIS] = '%s:%s' % (str(start), str(stop))
                        changed = True
                    elif axtype == pyferret.AXISTYPE_LEVEL:
                        if coordlimits[pyferret.Z_AXIS] or indexlimits[pyferret.Z_AXIS]:
                            raise KeyError('two level slices given')
                        coordlimits[pyferret.Z_AXIS] = '%s:%s' % (str(start), str(stop))
                        changed = True
                    elif axtype == pyferret.AXISTYPE_TIME:
                        if coordlimits[pyferret.T_AXIS] or indexlimits[pyferret.T_AXIS]:
                            raise KeyError('two time slices given')
                        starttime = '"%02d-%3s-%04d %02d:%02d:%02d"' % \
                            ( start[pyferret.TIMEARRAY_DAYINDEX],
                              pyferret._UC_MONTH_NAMES[start[pyferret.TIMEARRAY_MONTHINDEX]],
                              start[pyferret.TIMEARRAY_YEARINDEX],
                              start[pyferret.TIMEARRAY_HOURINDEX],
                              start[pyferret.TIMEARRAY_MINUTEINDEX],
                              start[pyferret.TIMEARRAY_SECONDINDEX] )
                        stoptime = '"%02d-%3s-%04d %02d:%02d:%02d"' % \
                            ( stop[pyferret.TIMEARRAY_DAYINDEX],
                              pyferret._UC_MONTH_NAMES[stop[pyferret.TIMEARRAY_MONTHINDEX]],
                              stop[pyferret.TIMEARRAY_YEARINDEX],
                              stop[pyferret.TIMEARRAY_HOURINDEX],
                              stop[pyferret.TIMEARRAY_MINUTEINDEX],
                              stop[pyferret.TIMEARRAY_SECONDINDEX] )
                        coordlimits[pyferret.T_AXIS] = '%s:%s' % (starttime, stoptime)
                        changed = True
                    elif isinstance(start,int) and isinstance(stop,int):
                        if coordlimits[k] or indexlimits[k]:
                            raise KeyError('two slices for axis index %d given' % k)
                        indexlimits[k] = '%d:%d' % (start, stop)
                        changed = True
                    elif isinstance(start,numbers.Real) and isinstance(stop,numbers.Real):
                        if coordlimits[k] or indexlimits[k]:
                            raise KeyError('two slices for axis index %d given' % k)
                        coordlimits[k] = '%s:%s' % (str(start), str(stop))
                        changed = True
                    else:
                        raise KeyError('%s in not valid' % str(piece))
                else:
                    try:
                        (axtype, val) = pyferret.FerrGrid._parsegeoval(piece)
                    except Exception as ex:
                        raise KeyError('%s is not valid: %s' % (str(piece), str(ex)))
                    if axtype == pyferret.AXISTYPE_LONGITUDE:
                        if coordlimits[pyferret.X_AXIS] or indexlimits[pyferret.X_AXIS]:
                            raise KeyError('two longitude slices given')
                        coordlimits[pyferret.X_AXIS] = '%s' % str(val)
                        changed = True
                    elif axtype == pyferret.AXISTYPE_LATITUDE:
                        if coordlimits[pyferret.Y_AXIS] or indexlimits[pyferret.Y_AXIS]:
                            raise KeyError('two latitude slices given')
                        coordlimits[pyferret.Y_AXIS] = '%s' % str(val)
                        changed = True
                    elif axtype == pyferret.AXISTYPE_LEVEL:
                        if coordlimits[pyferret.Z_AXIS] or indexlimits[pyferret.Z_AXIS]:
                            raise KeyError('two level slices given')
                        coordlimits[pyferret.Z_AXIS] = '%s' % str(val)
                        changed = True
                    elif axtype == pyferret.AXISTYPE_TIME:
                        if coordlimits[pyferret.T_AXIS] or indexlimits[pyferret.T_AXIS]:
                            raise KeyError('two time slices given')
                        coordlimits[pyferret.T_AXIS] = '"%02d-%3s-%04d %02d:%02d:%02d"' % \
                            ( val[pyferret.TIMEARRAY_DAYINDEX],
                              pyferret._UC_MONTH_NAMES[val[pyferret.TIMEARRAY_MONTHINDEX]],
                              val[pyferret.TIMEARRAY_YEARINDEX],
                              val[pyferret.TIMEARRAY_HOURINDEX],
                              val[pyferret.TIMEARRAY_MINUTEINDEX],
                              val[pyferret.TIMEARRAY_SECONDINDEX] )
                        changed = True
                    elif isinstance(val,int):
                        if coordlimits[k] or indexlimits[k]:
                            raise KeyError('two slices for axis index %d given' % k)
                        indexlimits[k] = '%d' % val
                        changed = True
                    elif isinstance(start,float):
                        if coordlimits[k] or indexlimits[k]:
                            raise KeyError('two slices for axis index %d given' % k)
                        coordlimits[k] = '%s' % str(val)
                        changed = True
                    else:
                        raise KeyError('%s in not valid' % str(piece))
        elif isinstance(key, slice):
            try:
                (axtype, start, stop, step) = pyferret.FerrGrid._parsegeoslice(key)
            except Exception as ex:
                raise KeyError('%s is not valid: %s' % (str(key), str(ex)))
            if axtype == pyferret.AXISTYPE_LONGITUDE:
                coordlimits[pyferret.X_AXIS] = '%s:%s' % (str(start), str(stop))
                changed = True
            elif axtype == pyferret.AXISTYPE_LATITUDE:
                coordlimits[pyferret.Y_AXIS] = '%s:%s' % (str(start), str(stop))
                changed = True
            elif axtype == pyferret.AXISTYPE_LEVEL:
                coordlimits[pyferret.Z_AXIS] = '%s:%s' % (str(start), str(stop))
                changed = True
            elif axtype == pyferret.AXISTYPE_TIME:
                starttime = '"%02d-%3s-%04d %02d:%02d:%02d"' % \
                    ( start[pyferret.TIMEARRAY_DAYINDEX],
                      pyferret._UC_MONTH_NAMES[start[pyferret.TIMEARRAY_MONTHINDEX]],
                      start[pyferret.TIMEARRAY_YEARINDEX],
                      start[pyferret.TIMEARRAY_HOURINDEX],
                      start[pyferret.TIMEARRAY_MINUTEINDEX],
                      start[pyferret.TIMEARRAY_SECONDINDEX] )
                stoptime = '"%02d-%3s-%04d %02d:%02d:%02d"' % \
                    ( stop[pyferret.TIMEARRAY_DAYINDEX],
                      pyferret._UC_MONTH_NAMES[stop[pyferret.TIMEARRAY_MONTHINDEX]],
                      stop[pyferret.TIMEARRAY_YEARINDEX],
                      stop[pyferret.TIMEARRAY_HOURINDEX],
                      stop[pyferret.TIMEARRAY_MINUTEINDEX],
                      stop[pyferret.TIMEARRAY_SECONDINDEX] )
                coordlimits[pyferret.T_AXIS] = '%s:%s' % (starttime, stoptime)
                changed = True
            elif isinstance(start,int) and isinstance(stop,int):
                indexlimits[0] = '%d:%d' % (start, stop)
                changed = True
            elif isinstance(start,numbers.Real) and isinstance(stop,numbers.Real):
                coordlimits[0] = '%s:%s' % (str(start), str(stop))
                changed = True
            else:
                raise KeyError('%s in not valid' % str(key))
        else:
            try:
                (axtype, val) = pyferret.FerrGrid._parsegeoval(key)
            except Exception as ex:
                raise KeyError('%s is not valid: %s' % (str(key), str(ex)))
            if axtype == pyferret.AXISTYPE_LONGITUDE:
                coordlimits[pyferret.X_AXIS] = '%s' % str(val)
                changed = True
            elif axtype == pyferret.AXISTYPE_LATITUDE:
                coordlimits[pyferret.Y_AXIS] = '%s' % str(val)
                changed = True
            elif axtype == pyferret.AXISTYPE_LEVEL:
                coordlimits[pyferret.Z_AXIS] = '%s' % str(val)
                changed = True
            elif axtype == pyferret.AXISTYPE_TIME:
                coordlimits[pyferret.T_AXIS] = '"%02d-%3s-%04d %02d:%02d:%02d"' % \
                    ( val[pyferret.TIMEARRAY_DAYINDEX],
                      pyferret._UC_MONTH_NAMES[val[pyferret.TIMEARRAY_MONTHINDEX]],
                      val[pyferret.TIMEARRAY_YEARINDEX],
                      val[pyferret.TIMEARRAY_HOURINDEX],
                      val[pyferret.TIMEARRAY_MINUTEINDEX],
                      val[pyferret.TIMEARRAY_SECONDINDEX] )
                changed = True
            elif isinstance(val,int):
                indexlimits[k] = '%d' % val
                changed = True
            elif isinstance(start,float):
                coordlimits[k] = '%s' % str(val)
                changed = True
            else:
                raise KeyError('%s in not valid' % str(key))
        if not changed:
            # the whole thing - definition is just this variable
            newvar = FerrVar(definition=self.ferretname())
            newvar._requires.update(self._requires)
            return newvar
        # create the subset definition in Ferret
        if self._datasetname:
            newdef = '%s[d=%s,' % (self._varname, self._datasetname)
        else:
            newdef = '%s[' % self._varname
        if coordlimits[pyferret.X_AXIS]:
            newdef += 'X=%s,' % coordlimits[pyferret.X_AXIS]
        if indexlimits[pyferret.X_AXIS]:
            newdef += 'I=%s,' % indexlimits[pyferret.X_AXIS]
        if coordlimits[pyferret.Y_AXIS]:
            newdef += 'Y=%s,' % coordlimits[pyferret.Y_AXIS]
        if indexlimits[pyferret.Y_AXIS]:
            newdef += 'J=%s,' % indexlimits[pyferret.Y_AXIS]
        if coordlimits[pyferret.Z_AXIS]:
            newdef += 'Z=%s,' % coordlimits[pyferret.Z_AXIS]
        if indexlimits[pyferret.Z_AXIS]:
            newdef += 'K=%s,' % indexlimits[pyferret.Z_AXIS]
        if coordlimits[pyferret.T_AXIS]:
            newdef += 'T=%s,' % coordlimits[pyferret.T_AXIS]
        if indexlimits[pyferret.T_AXIS]:
            newdef += 'L=%s,' % indexlimits[pyferret.T_AXIS]
        if coordlimits[pyferret.E_AXIS]:
            newdef += 'E=%s,' % coordlimits[pyferret.E_AXIS]
        if indexlimits[pyferret.E_AXIS]:
            newdef += 'M=%s,' % indexlimits[pyferret.E_AXIS]
        if coordlimits[pyferret.F_AXIS]:
            newdef += 'F=%s,' % coordlimits[pyferret.F_AXIS]
        if indexlimits[pyferret.F_AXIS]:
            newdef += 'N=%s,' % indexlimits[pyferret.F_AXIS]
        # replace the final , with ]
        newdef = newdef[:-1] + ']'
        newvar = FerrVar(definition=newdef)
        newvar._requires.update(self._requires)
        return newvar

    def assign(self, varname, datasetname):
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
        if varname.upper() in self._requires:
            raise ValueError('recursive definitions cannot be implemented in Ferret')
        # Assign the variable in Ferret
        if datasetname:
            cmdstr = 'DEFINE VAR /d=%s %s = %s' % (datasetname, varname, self._definition)
        else:
            cmdstr = 'DEFINE VAR %s = %s' % (varname, self._definition)
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('problems defining %s (%s) in Ferret: %s' % (varname, cmdstr, errmsg))
        # Revise the fields in this FerrVar to reflect this assignment
        self._varname = varname
        if datasetname:
            self._datasetname = datasetname
        else:
            self._datasetname = ''
        self._definition = self.ferretname()
        self._requires.add(varname.upper())
        self.clean()

    def remove(self):
        '''
        Removes (cancels) this variable in Ferret, then cleans this FerrVar and erases _varname.
        Raises a ValueError if there is a problem, such as if this is a file variable.
        '''
        ferrname = self.ferretname()
        if self._isfilevar:
            raise ValueError('cannot remove file variable %s from Ferret' % ferrname)
        cmdstr = 'CANCEL VAR %s' % ferrname
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('cannot remove variable %s from Ferret: %s' % (ferrname, errmsg))
        self._varname = ''
        self.clean()

    def clean(self):
        '''
        Clears the grid and data stored in this FerrVar.  After this call, any 
        request for the grid or data will automatically fetch the latest values 
        from Ferret.  This method should be called anytime there is a change 
        in the definition of this variable, or a variable this variable uses.
        '''
        self._datagrid = None
        self._dataarray = None
        self._dataunit = ''
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
        ferrname = self.ferretname()
        datadict = pyferret.getdata(ferrname, False)
        self._datagrid = pyferret.FerrGrid(gridname=ferrname,
                                           axistypes=datadict["axis_types"], 
                                           axiscoords=datadict["axis_coords"], 
                                           axisunits=datadict["axis_units"], 
                                           axisnames=datadict["axis_names"])
        self._dataarray = datadict["data"]
        self._dataunit = datadict["data_unit"]
        self._missingvalue = datadict["missing_value"]

    def showgrid(self, qual=''):
        '''
        Show the Ferret grid information about this variable.  This uses 
        the Ferret SHOW GRID command to create and display the information.
            qual (string): Ferret qualifiers to add to the SHOW GRID command
        '''
        if not isinstance(qual, str):
            raise ValueError('qual (Ferret qualifiers) must be a string')
        cmdstr = 'SHOW GRID'
        if qual:
            cmdstr += qual
        cmdstr += ' '
        cmdstr += self.ferretname()
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('Ferret command "%s" failed: %s' % (cmdstr, errmsg))

    def regrid(self, newgrid, method=REGRID_LINEAR):
        '''
        Returns an anonymous FerrVar that is this variable regridded to the grid
        implied by newgrid using the given method.
            newgrid (FerrVar |  string | FerrGrid): regrid to this implied grid;
                if a FerrVar, the implied grid is the grid used by the Ferret variable,
                if a string, the implied grid is the grid known to Ferret by this name
                if a FerrGrid, the implied grid is this grid (TODO: implement)
            method (string): method to perform the regridding; typically one of
                pyferret.REGRID_LINEAR (default)
                    (multi-axis) linear interpolation of nearest source points around destination point
                pyferret.REGRID_AVERAGE
                    length-weighted averaging of source point cells overlapping destination point cell
                pyferret.REGRID_ASSOCIATE
                    blind association of source points to destination points by indices
                pyferret.REGRID_MEAN
                    unweighted mean of source points in destination point cell
                pyferret.REGRID_NEAREST
                    value of source point nearest the destination point
                pyferret.REGRID_MIN
                    minimum value of source points in destination point cell 
                pyferret.REGRID_MAX
                    maximum value of source points in destination point cell 
                pyferret.REGRID_EXACT
                    copy values where source and destination points coincide; 
                    other destination points assigned missing value
        '''
        if not self._varname:
            raise NotImplementedError('regridding can only be performed on variables assigned in Ferret')
        if not ( isinstance(method, str) and (method[0] == '@') ):
            raise ValueError('invalid regridding method %s' % str(method))
        if isinstance(newgrid, FerrVar):
            if not newgrid._varname:
                raise ValueError('FerrVar used for the new grid is not assigned in Ferret')
            gridname = newgrid.ferretname()
        elif isinstance(newgrid, str):
            gridname = newgrid
        elif isinstance(newgrid, FerrGrid):
            raise NotImplementedError('regrid using FerrGrid not implemented at this time')
        if self._datasetname:
            newdef = '%s[d=%s,g=%s%s]' % (self._varname, self._datasetname, gridname, method)
        else:
            newdef = '%s[g=%s%s]' % (self._varname, gridname, method)
        newvar = FerrVar(definition=newdef)
        newvar._requires.update(self._requires)
        return newvar

