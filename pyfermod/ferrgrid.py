'''
Represents Ferret grids in Python.
'''

import numbers
import time
import numpy
import pyferret

# set of valid axis types
_VALID_AXIS_TYPES = frozenset( (pyferret.AXISTYPE_LONGITUDE, 
                                pyferret.AXISTYPE_LATITUDE, 
                                pyferret.AXISTYPE_LEVEL, 
                                pyferret.AXISTYPE_TIME, 
                                pyferret.AXISTYPE_CUSTOM, 
                                pyferret.AXISTYPE_ABSTRACT, 
                                pyferret.AXISTYPE_NORMAL) )
_VALID_AXIS_NUMS = frozenset( (pyferret.X_AXIS,
                               pyferret.Y_AXIS,
                               pyferret.Z_AXIS,
                               pyferret.T_AXIS,
                               pyferret.E_AXIS,
                               pyferret.F_AXIS) )

# Supported formats for time.strptime
_TIME_PARSE_FORMATS = ( 
    '%d-%b-%Y %H:%M:%S',
    '%d-%b-%Y %H:%M',
    '%d-%b-%Y',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%dT%H:%M',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
    '%Y-%m-%d', 
)
_TIME_NOYEAR_PARSE_FORMATS = (
    '%d-%b %H:%M:%S',
    '%d-%b',
)


class FerrGrid(object):
    '''
    Ferret grid object
    '''

    def __init__(self, gridname=None, axistypes=None, axiscoords=None, axisunits=None, axisnames=None):
        '''
        Describe a Ferret grid using the given information about the axes.
            gridname (string): Ferret name for the grid (or the variable using this grid)
            axistypes (sequence of int): types of the axes in the grid; valid values are
                    pyferret.AXISTYPE_LONGITUDE
                    pyferret.AXISTYPE_LATITUDE
                    pyferret.AXISTYPE_LEVEL
                    pyferret.AXISTYPE_TIME
                    pyferret.AXISTYPE_CUSTOM   (axis units not recognized by Ferret)
                    pyferret.AXISTYPE_ABSTRACT (axis is unit-less integer values)
                    pyferret.AXISTYPE_NORMAL   (axis is normal to the data)
            axiscoords (sequence of sequence of numeric): coordinate values of each axis; 
                for axes that are neither a time axis nor normal to the data, a 1-D array
                of numeric values; for time axes, an (n,6) 2D array of integers where 
                each time step is formed from the six integers for the day, month, year, 
                hour, minute, and second in the index given by
                    pyferret.TIMEARRAY_DAYINDEX
                    pyferret.TIMEARRAY_MONTHINDEX
                    pyferret.TIMEARRAY_YEARINDEX
                    pyferret.TIMEARRAY_HOURINDEX
                    pyferret.TIMEARRAY_MINUTEINDEX
                    pyferret.TIMEARRAY_SECONDINDEX
                (Thus, axiscoords[t, pyferret.TIMEARRAY_YEARINDEX] gives the year of time point t.)
                Note: a relative time axis will be of type AXISTYPE_CUSTOM, with a unit
                      indicating the starting point, such as 'days since 01-JAN-2000'
                For axes normal to the data, the value is ignored.
            axisunits (sequence of string): units of each axis; for a time axis, this gives
                the calendar as one of 
                    pyferret.CALTYPE_360DAY
                    pyferret.CALTYPE_NOLEAP
                    pyferret.CALTYPE_GREGORIAN
                    pyferret.CALTYPE_JULIAN
                    pyferret.CALTYPE_ALLLEAP
                    pyferret.CALTYPE_NONE    (calendar not specified)
                For axes normal to the data, the value is ignored.
            axisnames (sequence of string): Ferret name for each axis
        '''
        if gridname:
            if not isinstance(gridname, str): 
                raise TypeError('gridname is not a string')
            self._gridname = gridname
        else:
            self._gridname = ''
        # axis types
        self._axistypes = [ pyferret.AXISTYPE_NORMAL ] * pyferret.MAX_FERRET_NDIM
        if axistypes:
            try:
                for k in xrange(len(axistypes)):
                    axtype = axistypes[k]
                    if not axtype in _VALID_AXIS_TYPES:
                        raise ValueError('axis type %s is not valid' % str(axtype))
                    self._axistypes[k] = axtype
            except TypeError:
                raise TypeError('axistypes is not a sequence type')
            except IndexError:
                raise ValueError('more than %d axis types specified' % pyferret.MAX_FERRET_NDIM)
        # axis names
        self._axisnames = [ '' ] * pyferret.MAX_FERRET_NDIM
        if axisnames:
            try:
                for k in xrange(len(axisnames)):
                    if self._axistypes[k] != pyferret.AXISTYPE_NORMAL:
                        axname = axisnames[k]
                        if axname:
                            if not isinstance(axname, str): 
                                raise ValueError('axis name %s is not valid' % str(axname))
                            self._axisnames[k] = axname
            except TypeError:
                raise TypeError('axisnames is not a sequence type')
            except IndexError:
                raise ValueError('more than %d axis names specified' % pyferret.MAX_FERRET_NDIM)
        # axis units
        self._axisunits = [ '' ] * pyferret.MAX_FERRET_NDIM
        if axisunits:
            try:
                for k in xrange(len(axisunits)):
                    if self._axistypes[k] != pyferret.AXISTYPE_NORMAL:
                        axunit = axisunits[k]
                        if axunit:
                            if not isinstance(axunit, str): 
                                raise ValueError('axis unit %s is not valid' % str(axunit))
                            self._axisunits[k] = axunit
            except TypeError:
                raise TypeError('axisunits is not a sequence type')
            except IndexError:
                raise ValueError('more than %d axis units specified' % pyferret.MAX_FERRET_NDIM)
        # axis coordinates
	self._axiscoords = [ None ] * pyferret.MAX_FERRET_NDIM
        if axiscoords != None:
            try:
                for k in xrange(len(axiscoords)):
                    if self._axistypes[k] != pyferret.AXISTYPE_NORMAL:
                        axcoords = axiscoords[k]
                        if self._axistypes[k] == pyferret.AXISTYPE_TIME:
                            try:
                                self._axiscoords[k] = numpy.array(axcoords, dtype=numpy.int32, copy=True)
                            except ValueError:
                                raise ValueError('time axiscoords[%d] is not an integer array' % k)
                            if self._axiscoords[k].ndim != 2:
                                raise ValueError('time axiscoords[%d] is not a 2-D array' % k)
                            if self._axiscoords[k].shape[1] != 6:
                                raise ValueError('time axiscoords[%d] second dimension is not 6' % k)
                        else:
                            try:
                                self._axiscoords[k] = numpy.array(axcoords, dtype=numpy.float64, copy=True)
                            except ValueError:
                                raise ValueError('axiscoords[%d] is not a numeric array' % k)
                            if self._axiscoords[k].ndim != 1:
                                raise ValueError('axiscoords[%d] is not a 1-D array' % k)
            except TypeError:
                raise TypeError('axiscoords is not a sequence type')
            except IndexError:
                raise ValueError('more than %d axis coordinate arrays specified' % pyferret.MAX_FERRET_NDIM)


    def __repr__(self):
        '''
        Representation to recreate this FerrGrid
        '''
        # Not elegant, but will do
        spacer = ',\n         '
        infostr = "FerrGrid(gridname='" + self._gridname + "'" + \
                  spacer + 'axistype=' + repr(self._axistypes) + \
                  spacer + 'axiscoords=' + repr(self._axiscoords) + \
                  spacer + 'axisunits=' + repr(self._axisunits) + \
                  spacer + 'axisnames=' + repr(self._axisnames) + ')'
        return infostr


    def __eq__(self, other):
        '''
        Two FerrGrids are equal is all their contents are the same.  
        All string values are compared case-insensitive.
        '''
        if not isinstance(other, FerrGrid):
            return NotImplemented
        if self._gridname.upper() != other._gridname.upper():
            return False
        # _axistypes is a list of integers
        if self._axistypes != other._axistypes:
            return False
        # _axisnames is a list of strings
        for k in xrange(pyferret.MAX_FERRET_NDIM):
            if self._axisnames[k].upper() != other._axisnames[k].upper():
               return False
        # _axisunits is a list of strings
        for k in xrange(pyferret.MAX_FERRET_NDIM):
            if self._axisunits[k].upper() != other._axisunits[k].upper():
               return False
        # _axiscoords is a list of ndarray or None
        for k in xrange(pyferret.MAX_FERRET_NDIM):
            scoords = self._axiscoords[k]
            ocoords = other._axiscoords[k]
            if (scoords == None) and (ocoords == None):
                continue
            if (scoords == None) or (ocoords == None):
                return False
            if not numpy.allclose(scoords, ocoords):
                return False
        return True


    def __ne__(self, other):
        '''
        Two FerrGrids are not equal is any of their contents are not the same.  
        All string values are compared case-insensitive.
        '''
        if not isinstance(other, FerrGrid):
            return NotImplemented
        return not self.__eq__(other)


    def copy(self):
        '''
        Make a copy of this grid
        '''
        return self.modifiedcopy()


    def modifiedcopy(self, gridname=None, axis=None, axtype=None, axcoords=None, axunit=None, axname=None):
        '''
        Returns a copy of this grid with the given name.  If axis is given, 
        then the information in the new grid for this axis is replaced by the 
        information in the remaining arguments.
            gridname (string): Ferret name for the new grid; if None:
                if axis is None, the Ferret name of this grid is copied
                if axis in not None, no Ferret name is given (empty string)
            axis (int): if not None, index of the axis to modify; one of:
                    pyferret.X_AXIS (0)
                    pyferret.Y_AXIS (1)
                    pyferret.Z_AXIS (2)
                    pyferret.T_AXIS (3)
                    pyferret.E_AXIS (4)
                    pyferret.F_AXIS (5)
        If axis is None, the remaining arguments are ignored.
            axtype (int): type of the new axis; one of:
                    pyferret.AXISTYPE_LONGITUDE
                    pyferret.AXISTYPE_LATITUDE
                    pyferret.AXISTYPE_LEVEL
                    pyferret.AXISTYPE_TIME
                    pyferret.AXISTYPE_CUSTOM   (axis units not recognized by Ferret)
                    pyferret.AXISTYPE_ABSTRACT (axis is unit-less integer values)
                    pyferret.AXISTYPE_NORMAL   (axis is normal to the data)
        If axtype is pyferret.AXISTYPE_NORMAL, the remaining arguments are ignored.
            axcoords (sequence of numeric): coordinate values of the new axis;
                for an axis that is neither a time axis nor normal to the data, 
                    this is a 1-D array of numeric values; 
                for time axes, an (n,6) 2D array of integers where each time step 
                    is formed from the six integers for the day, month, year, hour, 
                    minute, and second in the index given by
                        pyferret.TIMEARRAY_DAYINDEX
                        pyferret.TIMEARRAY_MONTHINDEX
                        pyferret.TIMEARRAY_YEARINDEX
                        pyferret.TIMEARRAY_HOURINDEX
                        pyferret.TIMEARRAY_MINUTEINDEX
                        pyferret.TIMEARRAY_SECONDINDEX
                (Thus, axcoords[t, pyferret.TIMEARRAY_YEARINDEX] gives the year 
                of time point t.)
                Note: a relative time axis will be of type AXISTYPE_CUSTOM, with a unit
                      indicating the starting point, such as 'days since 01-JAN-2000'
            axisunit (string): unit of the new axis; 
                for a time axis, this gives the calendar as one of
                    pyferret.CALTYPE_360DAY
                    pyferret.CALTYPE_NOLEAP
                    pyferret.CALTYPE_GREGORIAN
                    pyferret.CALTYPE_JULIAN
                    pyferret.CALTYPE_ALLLEAP
                    pyferret.CALTYPE_NONE    (calendar not specified)
            axisnames (sequence of string): Ferret name for this axis
        '''
        if (gridname == None) and (axis == None):
            newgridname = self._gridname
        else:
            newgridname = gridname
        newgrid = FerrGrid(newgridname, 
                           axistypes=self._axistypes, 
                           axiscoords=self._axiscoords, 
                           axisunits=self._axisunits, 
                           axisnames=self._axisnames)
        if axis == None:
            # ignore the remaining arguments
            return newgrid
        if not axis in _VALID_AXIS_NUMS:
            raise ValueError('axis must one of the pyferret constants ' + \
                             'X_AXIS, Y_AXIS, Z_AXIS, T_AXIS, E_AXIS, or F_AXIS')
        if not axtype in _VALID_AXIS_TYPES:
            raise ValueError('axis type %s is not valid' % str(axtype))
        newgrid._axistypes[axis] = axtype
        # set default values for the new axis
        newgrid._axisnames[axis] = ''
        newgrid._axisunits[axis] = ''
        newgrid._axiscoords[axis] = None
        if axtype == pyferret.AXISTYPE_NORMAL:
            # ignore the remaining arguments
            return newgrid
        if axname:
            if not isinstance(axname, str):
                raise ValueError('axis name %s is not valid' % str(axname))
            newgrid._axisnames[axis] = axname
        if axunit:
            if not isinstance(axunit, str):
                raise ValueError('axis unit %s is not valid' % str(axunit))
            newgrid._axisunits[axis] = axunit
        if axcoords != None:
            if axtype == pyferret.AXISTYPE_TIME:
                try:
                    newgrid._axiscoords[axis] = numpy.array(axcoords, dtype=numpy.int32, copy=True)
                except ValueError:
                    raise ValueError('(time) axcoords is not an integer array')
                if newgrid._axiscoords[axis].ndim != 2:
                    raise ValueError('(time) axcoords is not a 2-D array')
                if newgrid._axiscoords[axis].shape[1] != 6:
                    raise ValueError('(time) axcoords second dimension is not 6')
            else:
                try:
                    newgrid._axiscoords[axis] = numpy.array(axcoords, dtype=numpy.float64, copy=True)
                except ValueError:
                    raise ValueError('axcoords is not a numeric array')
                if newgrid._axiscoords[axis].ndim != 1:
                    raise ValueError('axcoords is not a 1-D array')
        return newgrid


    @staticmethod
    def _parsegeoslice(geoslice):
        '''
        Parses the contents of the slice attributes, interpreting any geo- or time-references
        and returns a tuple with the resulting interpreted axis type, start, stop, and step values.
           geoslice (slice): slice that can contain georeferences or time references
           returns (axistype, start, stop, step) where:
              axistype is one of:
                  pyferret.AXISTYPE_LONGITUDE  (longitude units detected)
                  pyferret.AXISTYPE_LATITUDE   (latitude units detected)
                  pyferret.AXISTYPE_LEVEL      (level units detected)
                  pyferret.AXISTYPE_TIME       (time units detected)
                  pyferret.AXISTYPE_ABSTRACT   (no units)
              start, stop, and step are:
                  None if the correspond geoslice attribute is not given; otherwise,
                  a list of six numbers if axistype is pyferret.AXISTYPE_TIME, or
                  a number if axistype is not pyferret.AXISTYPE_TIME
        The list of six numbers for time values are ordered according to the indices:
            pyferret.TIMEARRAY_DAYINDEX
            pyferret.TIMEARRAY_MONTHINDEX
            pyferret.TIMEARRAY_YEARINDEX
            pyferret.TIMEARRAY_HOURINDEX
            pyferret.TIMEARRAY_MINUTEINDEX
            pyferret.TIMEARRAY_SECONDINDEX
        For non-time values, the start, stop, and step values are int objects 
            if only if corresponding slice objects were int objects.  Thus, int 
            objects should be interpreted as axis indices and float objects 
            should be interpreted as axis values.
        Raises a ValueError if start and stop indicate different axes; i.e., 
            "10E":"20N" or 10:"20N" or 10:"20-JAN-2000", or if the value contain 
            unrecognized units.  If not a time slice, it is acceptable for step to 
            have no units even when start and stop do.  If a time slice, the step 
            must have a unit of y, d, h, m, or s, which corresponds to year, day, 
            hour, minute, or second; there is no month time step unit.
        Raises a TypeError if geoslice is not a slice or None, or if the values 
            in the slice are not None and cannot be interpreted.
        '''
        if geoslice == None:
            return (pyferret.AXISTYPE_ABSTRACT, None, None, None)
        if not isinstance(geoslice, slice):
            raise TypeError('not a slice object: %s' % repr(geoslice))
        (starttype, start) = FerrGrid._parsegeoval(geoslice.start)
        (stoptype, stop) = FerrGrid._parsegeoval(geoslice.stop)
        # start and stop types must match (so 10:"25E" also fails)
        if starttype != stoptype:
            raise ValueError('mismatch of units: %s and %s' % (geoslice.start, geoslice.stop))
        axtype = starttype
        if axtype == pyferret.AXISTYPE_TIME:
            (steptype, step) = FerrGrid._parsegeoval(geoslice.step, istimestep=True)
            if (step != None) and (steptype != pyferret.AXISTYPE_TIME):
               raise ValueError('a time unit y, d, h, m, or s must be given with time slice steps')
        else:
            (steptype, step) = FerrGrid._parsegeoval(geoslice.step)
            if (steptype != pyferret.AXISTYPE_ABSTRACT) and (steptype != axtype):
               raise ValueError('mismatch of units: %s, %s' % (geoslice.start, geoslice.step))
            
        return (axtype, start, stop, step)

    @staticmethod
    def _parsegeoval(val, istimestep=False):
        '''
        Parses the value as either a longitude, latitude, level, time, or abstract number.
        If val is a numeric value, the tuple (pyferret.AXISTYPE_ABSTRACT, val) is returned.
        If val is None, the tuple (pyferret.AXISTYPE_ABSTRACT, None) is returned.
        If val is a longitude string (unit E or W when istimestep is false), 
            (pyferret.AXISTYPE_LONGITUDE, fval) is returned where fval 
            is the floating point longitude value.
        If val is a latitude string (unit N or S when istimestep is false), 
            (pyferret.AXISTYPE_LATITUDE, fval) is returned where fval 
            is the floating point latitude value.
        If val is a level string (unit m when istimestep is False), 
            (pyferret.AXISTYPE_LEVEL, fval) is returned where fval 
            is the floating point level value.
        If val is a date and, optionally, time string matching one of the formats given
            in _TIME_PARSE_FORMATS or _TIME_NOYEAR_PARSE_FORMATS, 
            (pyferret.AXISTYPE_TIME, tval) is returned where
            tval is a list of six numbers ordered by the indices:
                pyferret.TIMEARRAY_DAYINDEX
                pyferret.TIMEARRAY_MONTHINDEX
                pyferret.TIMEARRAY_YEARINDEX
                pyferret.TIMEARRAY_HOURINDEX
                pyferret.TIMEARRAY_MINUTEINDEX
                pyferret.TIMEARRAY_SECONDINDEX
        If istimestep is true and val is a time step string (unit y, d, h, m, or s),
            (pyferret.AXISTYPE_TIME, tval) is returned where tval is a list of six values 
            ordered by the above TIMEARRAY indices.  
            Note that m is minutes; there is no month timestep.
        If val is a string of a unitless number, (pyferret.AXISTYPE_ABSTACT, fval) is 
            returned where fval is the floating point value specified by val.
        If val is not numeric or a string, a TypeError is raised.
        If val is a string that cannot be parsed, a ValueError is raised.
        '''
        # if just a number, return it with abstract axis type
        if isinstance(val, numbers.Real):
            return (pyferret.AXISTYPE_ABSTRACT, val)
        # if None or empty, return None with abstract axis type
        if not val:
            return (pyferret.AXISTYPE_ABSTRACT, None)
        if not isinstance(val, str):
            raise TypeError('not a string: %s' % repr(val))
        if not istimestep:
            # first try parsing as a date/time string using the accepted formats
            for fmt in _TIME_PARSE_FORMATS:
                try:
                    tval = time.strptime(val, fmt)
                    tlist = [ 0, 0, 0, 0, 0, 0 ]
                    tlist[pyferret.TIMEARRAY_DAYINDEX] = tval.tm_mday
                    tlist[pyferret.TIMEARRAY_MONTHINDEX] = tval.tm_mon
                    tlist[pyferret.TIMEARRAY_YEARINDEX] = tval.tm_year
                    tlist[pyferret.TIMEARRAY_HOURINDEX] = tval.tm_hour
                    tlist[pyferret.TIMEARRAY_MINUTEINDEX] = tval.tm_min
                    tlist[pyferret.TIMEARRAY_SECONDINDEX] = tval.tm_sec
                    return (pyferret.AXISTYPE_TIME, tlist)
                except ValueError:
                    pass
            for fmt in _TIME_NOYEAR_PARSE_FORMATS:
                try:
                    tval = time.strptime(val, fmt)
                    tlist = [ 0, 0, 0, 0, 0, 0 ]
                    tlist[pyferret.TIMEARRAY_DAYINDEX] = tval.tm_mday
                    tlist[pyferret.TIMEARRAY_MONTHINDEX] = tval.tm_mon
                    # leave the year as zero - time assigns 1900
                    tlist[pyferret.TIMEARRAY_HOURINDEX] = tval.tm_hour
                    tlist[pyferret.TIMEARRAY_MINUTEINDEX] = tval.tm_min
                    tlist[pyferret.TIMEARRAY_SECONDINDEX] = tval.tm_sec
                    return (pyferret.AXISTYPE_TIME, tlist)
                except ValueError:
                    pass
        # not a date/time, so parse as a number with possibly a final letter for the unit
        try:
            lastchar = val[-1].upper()
            if (not istimestep) and (lastchar == 'E'): # degrees E
                fval = float(val[:-1])
                return(pyferret.AXISTYPE_LONGITUDE, fval)
            elif (not istimestep) and (lastchar == 'W'): # degrees W
                fval = -1.0 * float(val[:-1])
                return(pyferret.AXISTYPE_LONGITUDE, fval)
            elif (not istimestep) and (lastchar == 'N'): # degrees N
                fval = float(val[:-1])
                return(pyferret.AXISTYPE_LATITUDE, fval)
            elif (not istimestep) and (lastchar == 'S'): # degrees S
                fval = -1.0 * float(val[:-1])
                return(pyferret.AXISTYPE_LATITUDE, fval)
            elif (not istimestep) and (lastchar == 'M'): # meters
                fval = float(val[:-1])
                return(pyferret.AXISTYPE_LEVEL, fval)
            elif istimestep and (lastchar == 'Y'): # years
                fval = float(val[:-1])
                tlist = [ 0, 0, 0, 0, 0, 0 ]
                tlist[pyferret.TIMEARRAY_YEARINDEX] = fval
                return (pyferret.AXISTYPE_TIME, tlist)
            elif istimestep and (lastchar == 'D'): # days
                fval = float(val[:-1])
                tlist = [ 0, 0, 0, 0, 0, 0 ]
                tlist[pyferret.TIMEARRAY_DAYINDEX] = fval
                return (pyferret.AXISTYPE_TIME, tlist)
            elif istimestep and (lastchar == 'H'): # hours
                fval = float(val[:-1])
                tlist = [ 0, 0, 0, 0, 0, 0 ]
                tlist[pyferret.TIMEARRAY_HOURINDEX] = fval
                return (pyferret.AXISTYPE_TIME, tlist)
            elif istimestep and (lastchar == 'M'): # minutes
                fval = float(val[:-1])
                tlist = [ 0, 0, 0, 0, 0, 0 ]
                tlist[pyferret.TIMEARRAY_MINUTEINDEX] = fval
                return (pyferret.AXISTYPE_TIME, tlist)
            elif istimestep and (lastchar == 'S'): # seconds
                fval = float(val[:-1])
                tlist = [ 0, 0, 0, 0, 0, 0 ]
                tlist[pyferret.TIMEARRAY_SECONDINDEX] = fval
                return (pyferret.AXISTYPE_TIME, tlist)
            else:
                # maybe just numeric string; if not, will raise an exception
                fval = float(val)
                return(pyferret.AXISTYPE_ABSTRACT, fval)
        except Exception:
            raise ValueError('unable to parse: %s' % val)

    @staticmethod
    def _makedatestring(timearray):
        '''
        Creates a date and time string for the format DD-MON-YYYY HH:MM:SS 
        corresponding the values in the given time array.  If the year is 
        zero, -YYYY is omitted.  If the seconds is zero, :SS is omitted; 
        if hours, minutes, and seconds are all zero, HH:MM:SS is omitted.
            timearray: tuple of six int with time values given by the indices
                pyferret.TIMEARRAY_DAYINDEX
                pyferret.TIMEARRAY_MONTHINDEX
                pyferret.TIMEARRAY_YEARINDEX
                pyferret.TIMEARRAY_HOURINDEX
                pyferret.TIMEARRAY_MINUTEINDEX
                pyferret.TIMEARRAY_SECONDINDEX
        '''
        day = timearray[pyferret.TIMEARRAY_DAYINDEX]
        monthstr = pyferret.datamethods._UC_MONTH_NAMES[timearray[pyferret.TIMEARRAY_MONTHINDEX]]
        year = timearray[pyferret.TIMEARRAY_YEARINDEX]
        hour = timearray[pyferret.TIMEARRAY_HOURINDEX]
        minute = timearray[pyferret.TIMEARRAY_MINUTEINDEX]
        second = timearray[pyferret.TIMEARRAY_SECONDINDEX]
        if year > 0:
            datestr = '%02d-%3s-%04d' % (day, monthstr, year)
        else:
            datestr = '%02d-%3s' % (day, monthstr)
        if second > 0:
            timestr = ' %02d:%02d:%02d' % (hour, minute, second)
        elif (minute > 0) or (hour > 0):
            timestr = ' %02d:%02d' % (hour, minute)
        else:
            timestr = ''
        return datestr + timestr
 
