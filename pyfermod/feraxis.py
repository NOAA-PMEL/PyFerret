'''
Represents Ferret axes in Python.
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


class FerAxis(object):
    '''
    Ferret axis object
    '''

    def __init__(self, axtype=None, coords=None, unit=None, name=None):
        '''
        Describe a Ferret axis using the given information about the axis.
            axtype (int): type of the axis; valid values are
                    pyferret.AXISTYPE_LONGITUDE
                    pyferret.AXISTYPE_LATITUDE
                    pyferret.AXISTYPE_LEVEL
                    pyferret.AXISTYPE_TIME
                    pyferret.AXISTYPE_CUSTOM   (axis unit not recognized by Ferret)
                    pyferret.AXISTYPE_ABSTRACT (axis is unit-less integer values)
                    pyferret.AXISTYPE_NORMAL   (axis is normal to the data)
                if not given, AXISTYPE_NORMAL is used.
            coords (sequence of numeric): coordinate values of the axis; 
                for an axis that is neither a time axis, an abstract axis, nor normal 
                to the data, a 1-D array of numeric values; 
                for a time axis, an (n,6) 2D array of integers where each time step is 
                formed from the six integers for the day, month, year, hour, minute, 
                and second in the index given by
                    pyferret.TIMEARRAY_DAYINDEX
                    pyferret.TIMEARRAY_MONTHINDEX
                    pyferret.TIMEARRAY_YEARINDEX
                    pyferret.TIMEARRAY_HOURINDEX
                    pyferret.TIMEARRAY_MINUTEINDEX
                    pyferret.TIMEARRAY_SECONDINDEX
                (Thus, coords[t, TIMEARRAY_YEARINDEX] gives the year of time t.)
                Note: a relative time axis will be of type AXISTYPE_CUSTOM, with a unit
                      indicating the starting point, such as 'days since 01-JAN-2000'
                For an abstact axis or an axis normal to the data, this argument is ignored.
            unit (string): unit of the axis; for a time axis, this gives the calendar 
                    as one of
                    pyferret.CALTYPE_360DAY
                    pyferret.CALTYPE_NOLEAP
                    pyferret.CALTYPE_GREGORIAN
                    pyferret.CALTYPE_JULIAN
                    pyferret.CALTYPE_ALLLEAP
                    pyferret.CALTYPE_NONE    (calendar not specified)
                For abstact axes, or axes normal to the data, this argument is ignored.
            name (string): Ferret name for the axis
                For an axis normal to the data, this argument is ignored.
        '''
        # axis type
        if axtype:
            if not axtype in _VALID_AXIS_TYPES:
                raise ValueError('axis type %s is not valid' % str(axtype))
            self._axtype = axtype
        else:
            self._axtype = pyferret.AXISTYPE_NORMAL
        # axis name
        if name and (self._axtype != pyferret.AXISTYPE_NORMAL):
            if not isinstance(name, str): 
                raise ValueError('axis name %s is not valid' % str(name))
            self._name = name.strip()
        else:
            self._name = ''
        # axis unit
        if unit and (self._axtype != pyferret.AXISTYPE_NORMAL) \
                    and (self._axtype != pyferret.AXISTYPE_ABSTRACT):
            if not isinstance(unit, str): 
                raise ValueError('axis unit %s is not valid' % str(unit))
            self._unit = unit.strip()
        else:
            self._unit = ''
        # axis coordinates
        if (coords != None) and (self._axtype != pyferret.AXISTYPE_NORMAL) \
                            and (self._axtype != pyferret.AXISTYPE_ABSTRACT):
            if self._axtype == pyferret.AXISTYPE_TIME:
                try:
                    self._coords = numpy.array(coords, dtype=numpy.int32, copy=True)
                except ValueError:
                    raise ValueError('coordinates for a time axis is not an integer array')
                if self._coords.ndim != 2:
                    raise ValueError('coordinates for a time axis is not a 2-D array')
                if self._coords.shape[1] != 6:
                    raise ValueError('second dimenstion of coordinates for a time axis is not 6')
            else:
                try:
                    self._coords = numpy.array(coords, dtype=numpy.float64, copy=True)
                except ValueError:
                    raise ValueError('coordinates for an axis is not a numeric array')
                if self._coords.ndim != 1:
                    raise ValueError('coordinates for a lon/lat/level/custom axis is not a 1-D array' % k)
        else:
            self._coords = None


    def __repr__(self):
        '''
        Representation to recreate this FerAxis
        '''
        # Not elegant, but will do
        infostr = "FerAxis(axtype=" + repr(self._axtype) + \
                  ", coords=" + repr(self._coords) + \
                  ", unit='" + self._unit + \
                  "', name='" + self._name + "')"
        return infostr


    def __eq__(self, other):
        '''
        Two FerAxis objects are equal is all their contents are the same.  
        All string values are compared case-insensitive.
        '''
        if not isinstance(other, FerAxis):
            return NotImplemented
        # _axtype is an integer
        if self._axtype != other._axtype:
            return False
        # _name is a string
        if self._name.upper() != other._name.upper():
            return False
        # _unit is a string
        if self._unit.upper() != other._unit.upper():
            return False
        # _coords is an ndarray or None
        if (self._coords == None) and (other._coords == None):
            return True
        if (self._coords == None) or (other._coords == None):
            return False
        if not numpy.allclose(self._coords, other._coords):
            return False
        return True


    def __ne__(self, other):
        '''
        Two FerAxis obect are not equal is any of their contents are not 
        the same.  All string values are compared case-insensitive.
        '''
        if not isinstance(other, FerAxis):
            return NotImplemented
        return not self.__eq__(other)


    def copy(self):
        '''
        Returns a copy of this FerAxis object.  The FerAxis object returned
        does not share any mutable values (namely, the coordinates array)
        with this FerAxis object.
        '''
        # __init__ forces a copy of the coordinates array
        duplicate = FerAxis(axtype=self._axtype, coords=self._coords,
                            unit=self._unit, name=self._name)
        return duplicate


    def gettype(self):
        '''
        Returns the type of this axis as one of the integer constants
            pyferret.AXISTYPE_LONGITUDE
            pyferret.AXISTYPE_LATITUDE
            pyferret.AXISTYPE_LEVEL
            pyferret.AXISTYPE_TIME
            pyferret.AXISTYPE_CUSTOM   (axis unit not recognized by Ferret)
            pyferret.AXISTYPE_ABSTRACT (axis is unit-less integer values)
            pyferret.AXISTYPE_NORMAL   (axis is normal to the data)
        '''
        return self._axtype


    def getcoords(self):
        '''
        Returns a copy of the coordinates ndarray for this axis, 
        or None if there is no coordinates array for this axis.
        '''
        if self._coords != None:
            coords = self._coords.copy('A')
        else:
            coords = None
        return coords


    def getunit(self):
        '''
        Returns the unit string for this axis.  May be an empty string.
        '''
        return self._unit


    def getname(self):
        '''
        Returns the name string for this axis.  May be an empty string.
        '''
        return self._name


    @staticmethod
    def _parsegeoslice(geoslice):
        '''
        Parses the contents of the slice attributes, interpreting any geo- or time-references
        and returns a tuple with the resulting interpreted axis type, start, stop, and step values.

           geoslice (slice): slice that can contain georeferences or time references

           returns (axtype, start, stop, step) where:
              axtype is one of:
                  pyferret.AXISTYPE_LONGITUDE  (longitude units detected)
                  pyferret.AXISTYPE_LATITUDE   (latitude units detected)
                  pyferret.AXISTYPE_LEVEL      (level units detected)
                  pyferret.AXISTYPE_TIME       (time units detected)
                  pyferret.AXISTYPE_ABSTRACT   (no units)
              start, stop, and step are:
                  None if the correspond geoslice attribute is not given; otherwise,
                  a list of six numbers if axtype is pyferret.AXISTYPE_TIME, or
                  a number if axtype is not pyferret.AXISTYPE_TIME

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
        (starttype, start) = FerAxis._parsegeoval(geoslice.start)
        (stoptype, stop) = FerAxis._parsegeoval(geoslice.stop)
        # start and stop types must match (so 10:"25E" also fails)
        if starttype != stoptype:
            raise ValueError('mismatch of units: %s and %s' % (geoslice.start, geoslice.stop))
        axtype = starttype
        if axtype == pyferret.AXISTYPE_TIME:
            (steptype, step) = FerAxis._parsegeoval(geoslice.step, istimestep=True)
            if (step != None) and (steptype != pyferret.AXISTYPE_TIME):
               raise ValueError('a time unit y, d, h, m, or s must be given with time slice steps')
        else:
            (steptype, step) = FerAxis._parsegeoval(geoslice.step)
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
            # not a time *step* - first try parsing as a date/time string using the accepted formats
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
                    # leave the year as zero - time.strptime assigns 1900
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
                # make sure the rest is just numeric
                fval = float(val[:-1])
                return(pyferret.AXISTYPE_LONGITUDE, val.upper())
            elif (not istimestep) and (lastchar == 'W'): # degrees W
                # make sure the rest is just numeric
                fval = float(val[:-1])
                return(pyferret.AXISTYPE_LONGITUDE, val.upper())
            elif (not istimestep) and (lastchar == 'N'): # degrees N
                # make sure the rest is just numeric
                fval = float(val[:-1])
                return(pyferret.AXISTYPE_LATITUDE, val.upper())
            elif (not istimestep) and (lastchar == 'S'): # degrees S
                # make sure the rest is just numeric
                fval = float(val[:-1])
                return(pyferret.AXISTYPE_LATITUDE, val.upper())
            elif (not istimestep) and (lastchar == 'M'): # meters (or kilometers, etc.)
                return(pyferret.AXISTYPE_LEVEL, val.upper())
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
 

