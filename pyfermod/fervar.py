'''
Represents Ferret variables in Python.
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

_ADDED_ATTRIBUTES = ('data', 'grid', 'missval', 'unit')

class FerVar(object):
    '''
    Ferret variable object
    '''

    def __init__(self, defn=None, title=None):
        '''
        Creates a Ferret variable with (optionally) a title and a Ferret-syntax definition 
            defn (string): Ferret-syntax definition of the variable
            title (string): title (descriptive long name) for this variable
        '''
        # Record or generate the definition, or set to an empty string
        if defn:
            if not isinstance(defn, str):
                raise ValueError("defn is not a string")
            self._definition = defn
        else:
            self._definition = ''
        # Name of the variable in the dataset
        self._varname = ''
        # Name of the dataset
        self._dsetname = ''
        # Record the title for this variable, or am empty string if not given
        self.settitle(title)
        # Is this a file variable?
        self._isfilevar = False
        # The list of uppercase _varname's that are know to be used
        # in the definition.  This list is not guaranteed to be complete 
        # and is not used in comparisons.
        self._requires = set()
        # Call the unload method to create and set the defaults for 
        # _datagrid, _dataarray, _dataunit, and _missingvalue.
        #     _datagrid is a FerGrid describing the Ferret grid for the variable.
        #     _dataarray is a NumPy ndarray contains the Ferret data for the variable.
        #     _dataunit is a string given the unit of the data
        #     _missingvalue is the missing value used for the data
        self.unload()


    def copy(self):
        '''
        Return an anonymous copy (only the definition is copied) of this FerVar.
        '''
        newvar = FerVar(defn=self._definition)
        newvar._requires.update(self._requires)
        return newvar


    def settitle(self, title):
        '''
        Assigns the title (long descriptive name) for this FerVar.  If this
        variable is defined in Ferret, the title for the Ferret variable is
        also updated.
            title (string): title to assign
        Raises ValueError if title is not a string or if there is a problem 
            updating the title in Ferret
        '''
        if title:
            if not isinstance(title, str):
                raise ValueError("title is not a string")
            self._title = title
        else:
            self._title = ''
        if self._varname:
            cmdstr = 'SET VAR/TITLE="%s" %s' % (self._title, self.fername())
            (errval, errmsg) = pyferret.run(cmdstr)
            if errval != pyferret.FERR_OK:
                raise ValueError('problems updating the variable title in Ferret for ' + \
                                 '%s to "%s": %s' % (self.fername(), self._title, errmsg))
 

    def fername(self):
        ''' 
        Returns the Ferret name for this variable; namely,
            <_varname>[d=<_dsetname>]
        if _dsetname is given; otherwise, just
            <_varname>
        Raises ValueError if _varname is not defined
        '''
        if not self._varname:
            raise ValueError('this FerVar does not contain a Ferret variable name')
        if self._dsetname:
            fername = '%s[d=%s]' % (self._varname, self._dsetname)
        else:
            fername = '%s' % self._varname
        return fername


    def __repr__(self):
        '''
        Representation of this FerVar
        '''
        infostr = "FerVar(varname='%s', dsetname='%s', title = '%s', defn='%s')" \
                  % (self._varname, self._dsetname, self._title, self._definition)
        return infostr


    def __del__(self):
        '''
        Removes this variable, if possible, from Ferret.
        Any error are ignored.
        '''
        # Try to remove from Ferret but ignore errors
        try:
            self._removefromferret()
        except Exception:
            pass


    def __cmp__(self, other):
        '''
        FerVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, title, and 
        finally by the definition.  (Used by the "rich comparison" methods.)
        '''
        if not isinstance(other, FerVar):
            raise NotImplementedError('other is not a FerVar')
        supper = self._varname.upper()
        oupper = other._varname.upper()
        if supper < oupper:
            return -1
        if supper > oupper:
            return 1
        supper = self._dsetname.upper()
        oupper = other._dsetname.upper()
        if supper < oupper:
            return -1
        if supper > oupper:
            return 1
        supper = self._title.upper()
        oupper = other._title.upper()
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
        Two FerVars are equal if all of the following are True:
            they have the same Ferret variable name,
            they have the same dataset name, 
            they have the same title, and
            they have the same defintion.
        All these comparisons are case-insensitive.
        '''
        try:
            return ( self.__cmp__(other) == 0 )
        except NotImplementedError:
            return NotImplemented


    def __ne__(self, other):
        '''
        Two FerVars are not equal if any of the following are True:
            they have different Ferret variable name,
            they have different dataset name,
            they have different title, or
            they have different defintion.
        All these comparisons are case-insensitive.
        '''
        try:
            return ( self.__cmp__(other) != 0 )
        except NotImplementedError:
            return NotImplemented


    def __lt__(self, other):
        '''
        FerVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, title, and 
        finally by the definition.
        '''
        try:
            return ( self.__cmp__(other) < 0 )
        except NotImplementedError:
            return NotImplemented


    def __le__(self, other):
        '''
        FerVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, title, and 
        finally by the definition.
        '''
        try:
            return ( self.__cmp__(other) <= 0 )
        except NotImplementedError:
            return NotImplemented


    def __gt__(self, other):
        '''
        FerVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, title, and 
        finally by the definition.
        '''
        try:
            return ( self.__cmp__(other) > 0 )
        except NotImplementedError:
            return NotImplemented


    def __ge__(self, other):
        '''
        FerVars are ordered alphabetically, case-insensitive, first by 
        the Ferret variable name, then by the dataset name, title, and 
        finally by the definition.
        '''
        try:
            return ( self.__cmp__(other) >= 0 )
        except NotImplementedError:
            return NotImplemented


    def __nonzero__(self):
        '''
        Returns False if the Ferret variable name, dataset name, title, 
        and definition are all empty.  (For Python2.x)
        '''
        if self._varname:
            return True
        if self._dsetname:
            return True
        if self._title:
            return True
        if self._definition:
            return True
        return False


    def __bool__(self):
        '''
        Returns False if the Ferret variable name, dataset name, title 
        and definition are all empty.  (For Python3.x)
        '''
        return self.__nonzero__()


    def __add__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the sum of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the sum of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerVar):
            newdef = '(%s) + (%s)' % (self._definition, other._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) + %s' % (self._definition, str(other))
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented


    def __radd__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the sum of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the sum of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerVar):
            newdef = '(%s) + (%s)' % (other._definition, self._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s + (%s)' % (str(other), self._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented


    def __sub__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the difference (self - other) of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the difference (self - other) of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerVar):
            newdef = '(%s) - (%s)' % (self._definition, other._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) - %s' % (self._definition, str(other))
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented


    def __rsub__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the difference (other - self) of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the difference (other - self) of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerVar):
            newdef = '(%s) - (%s)' % (other._definition, self._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s - (%s)' % (str(other), self._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented


    def __mul__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the product of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the product of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerVar):
            newdef = '(%s) * (%s)' % (self._definition, other._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) * %s' % (self._definition, str(other))
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented


    def __rmul__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the product of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the product of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerVar):
            newdef = '(%s) * (%s)' % (other._definition, self._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s * (%s)' % (str(other), self._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented


    def __truediv__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the quotient (self / other) of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the quotient (self / other) of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        (For Python3.x)
        '''
        if isinstance(other, FerVar):
            newdef = '(%s) / (%s)' % (self._definition, other._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) / %s' % (self._definition, str(other))
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented


    def __rtruediv__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the quotient (other / self) of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the quotient (other / self) of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        (For Python3.x)
        '''
        if isinstance(other, FerVar):
            newdef = '(%s) / (%s)' % (other._definition, self._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s / (%s)' % (str(other), self._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented


    def __div__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the quotient (self / other) of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the quotient (self / other) of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        (For Python2.x)
        '''
        return self.__truediv__(other)


    def __rdiv__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the quotient (other / self) of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the quotient (other / self) of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        (For Python2.x)
        '''
        return self.__rtruediv__(other)


    def __pow__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the exponentiation (self ^ other) of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the exponentiation (self ^ other) of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerVar):
            newdef = '(%s) ^ (%s)' % (self._definition, other._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '(%s) ^ %s' % (self._definition, str(other))
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented


    def __rpow__(self, other):
        '''
        If other is a FerVar, returns an anonymous FerVar whose definition 
        is the exponentiation (other ^ self) of the FerVar definitions.
        If other is Real, returns an anonymous FerVar whose definition 
        is the exponentiation (other ^ self) of the FerVar definition with the number.
        If other is not a FerVar or Real, returns NotImplemented
        '''
        if isinstance(other, FerVar):
            newdef = '(%s) ^ (%s)' % (other._definition, self._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            newvar._requires.update(other._requires)
            return newvar
        if isinstance(other, numbers.Real):
            newdef = '%s ^ (%s)' % (str(other), self._definition)
            newvar = FerVar(defn=newdef)
            newvar._requires.update(self._requires)
            return newvar
        return NotImplemented


    def __neg__(self):
        '''
        Returns an anonymous FerVar whose definition is 
        the product of -1.0 and this FerVar definition.
        '''
        newdef = '-1.0 * (%s)' % self._definition
        newvar = FerVar(defn=newdef)
        newvar._requires.update(self._requires)
        return newvar


    def __pos__(self):
        '''
        Returns an anonymous FerVar whose definition is 
        the same as this FerVar definition.
        '''
        newvar = FerVar(defn=self._definition)
        newvar._requires.update(self._requires)
        return newvar


    def __abs__(self):
        '''
        Returns an anonymous FerVar whose definition is 
        the absolute value of this FerVar definition.
        '''
        newdef = 'abs(%s)' % self._definition
        newvar = FerVar(defn=newdef)
        newvar._requires.update(self._requires)
        return newvar


    def __getitem__(self, key):
        '''
        This FerVar must be assigned in Ferret.

        If key is 'data', returns the data array for this FerVar,
        loading it if necessary.
        If key is 'grid', returns the data grid for this FerVar,
        loading it if necessary.
        If key is 'missval', returns the value for missing data 
        for this FerVar.
        If key is 'unit', returns the data unit for this FerVar.

        Otherwise, assumes key is a slice or subset specification, 
        and returns an anonymous FerVar whose definition is a 
        subset of this FerVar.  
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
        if key is None:
            raise KeyError('None is not a valid key')
        if not self._varname:
            raise NotImplementedError('variable not assigned in Ferret')
        if key == 'data':
           return self.getdata()
        if key == 'grid':
           return self.getgrid()
        if key == 'missval':
           return self.getmissval()
        if key == 'unit':
           return self.getunit()

        coordlimits = [ None ] * pyferret.MAX_FERRET_NDIM
        indexlimits = [ None ] * pyferret.MAX_FERRET_NDIM
        changed = False
        # TODO: handle step values, try to condense code
        if isinstance(key, tuple):
            for k in range(len(key)):
                piece = key[k]
                if piece is None:
                    continue
                if isinstance(piece, slice):
                    try:
                        (axtype, start, stop, step) = pyferret.FerAxis._parsegeoslice(piece)
                    except Exception as ex:
                        raise KeyError('%s is not valid: %s' % (str(piece), str(ex)))
                    if step is not None:
                        raise KeyError('step values in slices are not supported at this time')
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
                        starttime = pyferret.FerAxis._makedatestring(start)
                        stoptime = pyferret.FerAxis._makedatestring(stop)
                        coordlimits[pyferret.T_AXIS] = '%s:%s' % (starttime, stoptime)
                        changed = True
                    elif isinstance(start,int) and isinstance(stop,int):
                        if coordlimits[k] or indexlimits[k]:
                            raise KeyError('two slices for axis index %d given' % k)
                        # do not know the axis length at this time
                        if (start < 0) or (stop < 0):
                            raise KeyError('negative indices not supported at this time')
                        # Ferret indices start at 1
                        start += 1
                        stop += 1
                        indexlimits[k] = '%d:%d' % (start, stop)
                        changed = True
                    elif isinstance(start,numbers.Real) and isinstance(stop,numbers.Real):
                        if coordlimits[k] or indexlimits[k]:
                            raise KeyError('two slices for axis index %d given' % k)
                        coordlimits[k] = '%s:%s' % (str(start), str(stop))
                        changed = True
                    elif (start is None) and (stop is None):
                        # full range on this axis 
                        if coordlimits[k] or indexlimits[k]:
                            raise KeyError('two slices for axis index %d given' % k)
                        continue
                    else:
                        raise KeyError('%s in not valid' % str(piece))
                else:
                    try:
                        (axtype, val) = pyferret.FerAxis._parsegeoval(piece)
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
                        coordlimits[pyferret.T_AXIS] = pyferret.FerAxis._makedatestring(val)
                        changed = True
                    elif isinstance(val,int):
                        if coordlimits[k] or indexlimits[k]:
                            raise KeyError('two slices for axis index %d given' % k)
                        # do not know the axis length at this time
                        if val < 0: 
                            raise KeyError('negative indices not supported at this time')
                        # Ferret indices start at 1
                        val += 1
                        indexlimits[k] = '%d' % val
                        changed = True
                    elif isinstance(val,numbers.Real):
                        if coordlimits[k] or indexlimits[k]:
                            raise KeyError('two slices for axis index %d given' % k)
                        coordlimits[k] = '%s' % str(val)
                        changed = True
                    else:
                        raise KeyError('%s in not valid' % str(piece))
        elif isinstance(key, slice):
            try:
                (axtype, start, stop, step) = pyferret.FerAxis._parsegeoslice(key)
            except Exception as ex:
                raise KeyError('%s is not valid: %s' % (str(key), str(ex)))
            if step is not None:
                raise KeyError('step values in slices are not supported at this time')
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
                starttime = pyferret.FerAxis._makedatestring(start)
                stoptime = pyferret.FerAxis._makedatestring(stop)
                coordlimits[pyferret.T_AXIS] = '%s:%s' % (starttime, stoptime)
                changed = True
            elif isinstance(start,int) and isinstance(stop,int):
                # do not know the axis length at this time
                if (start < 0) or (stop < 0):
                    raise KeyError('negative indices not supported at this time')
                # Ferret indices start at 1
                start += 1
                stop += 1
                indexlimits[0] = '%d:%d' % (start, stop)
                changed = True
            elif isinstance(start,numbers.Real) and isinstance(stop,numbers.Real):
                coordlimits[0] = '%s:%s' % (str(start), str(stop))
                changed = True
            elif (start is None) and (stop is None):
                # full range - standard way of generating a duplicate
                pass
            else:
                raise KeyError('%s in not valid' % str(key))
        else:
            try:
                (axtype, val) = pyferret.FerAxis._parsegeoval(key)
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
                coordlimits[pyferret.T_AXIS] = pyferret.FerAxis._makedatestring(val)
                changed = True
            elif isinstance(val,int):
                # do not know the axis length at this time
                if val < 0: 
                    raise KeyError('negative indices not supported at this time')
                # Ferret indices start at 1
                val += 1
                indexlimits[k] = '%d' % val
                changed = True
            elif isinstance(start,numbers.Real):
                coordlimits[k] = '%s' % str(val)
                changed = True
            else:
                raise KeyError('%s in not valid' % str(key))
        if not changed:
            # the whole thing - definition is just this variable
            newvar = FerVar(defn=self.fername())
            newvar._requires.update(self._requires)
            return newvar
        # create the subset definition in Ferret
        if self._dsetname:
            newdef = '%s[d=%s,' % (self._varname, self._dsetname)
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
        newvar = FerVar(defn=newdef)
        newvar._requires.update(self._requires)
        return newvar


    def __getattr__(self, name):
        '''
        Return the data array (if name='data'), data grid (if name='grid'), 
        name (if name='name'), or a copy of the coordinates (if name='coords')
        Note that this method is only called when the parent object 
        does not have an attribute with this name.
        '''
        try:
            if name in _ADDED_ATTRIBUTES:
                return self.__getitem__(name)
        except KeyError:
            pass
        raise AttributeError("unknown attribute '%s'" % name)


    def __dir__(self):
        '''
        Returns a list of known attributes, including those added 
        by the __getattr__ method.
        '''
        mydir = list(_ADDED_ATTRIBUTES)
        mydir.extend( dir(super(FerVar, self)) )
        return mydir


    def _markasknownvar(self, varname, dsetname, isfilevar):
        '''
        Marks this variable as a variable already defined in Ferret.
        '''
        if not varname:
            raise ValueError('varname is not given')
        if not isinstance(varname, str):
            raise ValueError('varname is not a string')
        if dsetname and not isinstance(varname, str):
            raise ValueError('dsetname name is not a string')
        self._varname = varname
        if dsetname:
            self._dsetname = dsetname
        else:
            self._dsetname = ''
        self._isfilevar = bool(isfilevar)
        self._definition = self.fername()
        self._requires.add(varname.upper())
        self.unload()


    def _assigninferret(self, varname, dsetname):
        '''
        Defines this FerVar in Ferret using the given variable name 
        associated with the given dataset name.
            varname (string): name for the variable in Ferret
            dsetname (string): name of the dataset to contain the variable
        Raises a ValueError if there is a problem.
        '''
        if not self._definition:
            raise ValueError('this FerVar does not contain a definition')
        if not varname:
            raise ValueError('variable name to be assigned is not given')
        if varname.upper() in self._requires:
            raise ValueError('recursive definitions cannot be implemented in Ferret')
        # Assign the variable in Ferret
        cmdstr = 'DEFINE VAR'
        if dsetname:
            cmdstr += '/D="%s"' % dsetname
        if self._title:
            cmdstr += '/TITLE="%s"' % self._title
        cmdstr += ' %s = %s' % (varname, self._definition)
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('problems defining %s (%s) in Ferret: %s' % (varname, cmdstr, errmsg))
        # Revise the fields in this FerVar to reflect this assignment
        self._markasknownvar(varname, dsetname, False)


    def _removefromferret(self):
        '''
        Removes (cancels) this variable in Ferret, then unloads this FerVar 
        and erases _varname.  Raises a NotImplementedError is this is a file 
        variable.  Raises a ValueError if there is a Ferret problem.  This 
        normally is not called by the user; instead delete the FerVar from 
        the dataset.
        '''
        # ignore if this Ferrer variable has already been removed from Ferret
        if not self._varname:
            return
        fername = self.fername()
        if self._isfilevar:
            raise NotImplementedError('%s is a file variable; close the dataset to remove' % fername)
        cmdstr = 'CANCEL VAR %s' % fername
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('unable to remove variable %s from Ferret: %s' % (fername, errmsg))
        self._varname = ''
        self.unload()


    def unload(self):
        '''
        Clears the grid and data stored in this FerVar.  After this call, any 
        request for the grid or data will automatically load the latest values 
        from Ferret.  This method should be called anytime there is a change 
        in the definition of this variable, or a variable this variable uses.
        '''
        self._datagrid = None
        self._dataarray = None
        self._dataunit = ''
        self._missingvalue = None


    def load(self):
        '''
        Retrieves the grid and data for this Ferret variable from Ferret.
        This method is automatically called before returning the grid or data 
        for the first time for this variable.  This can be called to update
        the grid or data in this FerVar after any change in the definition 
        of the variable.  Alternatively, cleardata can be called to clear any
        stored grid and data, delaying the update from Ferret until the grid
        or data is requested.
        Raises a ValueEror if problems occur.
        '''
        fername = self.fername()
        datadict = pyferret.getdata(fername, False)
        feraxes = [ ]
        for (axistype,axcoords,axunit,axname) in zip(
                datadict["axis_types"], datadict["axis_coords"], 
                datadict["axis_units"], datadict["axis_names"]):
            feraxes.append( pyferret.FerAxis(coords=axcoords, 
                    axtype=axistype, unit=axunit, name=axname) )
        self._datagrid = pyferret.FerGrid(axes=feraxes, name=fername)
        self._dataarray = datadict["data"]
        self._dataunit = datadict["data_unit"]
        self._missingvalue = datadict["missing_value"]


    def getdata(self):
        '''
        Returns a copy of the data array for this Ferret variable,
        first loading this variable if necessary.
        Raises a ValueError is a problem occurs.
        '''
        if (self._datagrid is None) or (self._dataarray is None):
            self.load()
        return self._dataarray.copy('A')


    def getgrid(self):
        '''
        Returns a copy of the data grid for this Ferret variable,
        first loading this variable if necessary.
        Raises a ValueError is a problem occurs.
        '''
        if (self._datagrid is None) or (self._dataarray is None):
            self.load()
        return self._datagrid.copy()


    def getmissval(self):
        '''
        Returns the value used for missing data for this Ferret 
        variable, first loading this variable if necessary.  
        Raises a ValueError is a problem occurs.
        '''
        if (self._datagrid is None) or (self._dataarray is None):
            self.load()
        # The missing value is a single-element ndarray
        return self._missingvalue[0]


    def getunit(self):
        '''
        Returns the unit string of the data for this Ferret
        variable, first loading this variable if necessary.
        Raises a ValueError is a problem occurs.
        '''
        if (self._datagrid is None) or (self._dataarray is None):
            self.load()
        return self._dataunit


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
        cmdstr += self.fername()
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('Ferret command "%s" failed: %s' % (cmdstr, errmsg))


    def regrid(self, newgrid, method=REGRID_LINEAR):
        '''
        Returns an anonymous FerVar that is this variable regridded to the grid
        implied by newgrid using the given method.
            newgrid (FerVar |  string | FerGrid): regrid to this implied grid;
                if a FerVar, the implied grid is the grid used by the Ferret variable,
                if a string, the implied grid is the grid known to Ferret by this name
                if a FerGrid, the implied grid is this grid (TODO: implement)
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
        if isinstance(newgrid, FerVar):
            if not newgrid._varname:
                raise ValueError('FerVar used for the new grid is not assigned in Ferret')
            gridname = newgrid.fername()
        elif isinstance(newgrid, str):
            gridname = newgrid
        elif isinstance(newgrid, pyferret.FerGrid):
            raise NotImplementedError('regrid using FerGrid not implemented at this time')
        if self._dsetname:
            newdef = '%s[d=%s,g=%s%s]' % (self._varname, self._dsetname, gridname, method)
        else:
            newdef = '%s[g=%s%s]' % (self._varname, gridname, method)
        newvar = FerVar(defn=newdef)
        newvar._requires.update(self._requires)
        return newvar

