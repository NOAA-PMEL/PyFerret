'''
Represents Ferret grids in Python.

@author: Karl Smith
'''

import numpy
import pyferret

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
                for axes that are neither a time axis nor normal to the data, an 1-D array
                of numeric values; for time axes, an (n,6) 2D array of integers where 
                each time step is formed from the six integers for the day, month, year, 
                hour, minute, and second in the index given by
                    pyferret.TIMEARRAY_DAYINDEX
                    pyferret.TIMEARRAY_MONTHINDEX
                    pyferret.TIMEARRAY_YEARINDEX
                    pyferret.TIMEARRAY_HOURINDEX
                    pyferret.TIMEARRAY_MINUTEINDEX
                    pyferret.TIMEARRAY_SECONDINDEX
                (Thus, axis_coords[t, pyferret.TIMEARRAY_YEARINDEX] gives the year of time point t.)
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
        self._gridname = gridname
        # axis types
        self._axistypes = [ pyferret.AXISTYPE_NORMAL ] * pyferret.MAX_FERRET_NDIM
        if axistypes:
            try:
                for k in xrange(len(axistypes)):
                    axtype = axistypes[k]
                    if not axtype in pyferret.VALID_AXIS_TYPES:
                        raise ValueError('axis type %s is not valid' % str(axtype))
                    self._axistypes[k] = axtype
            except TypeError:
                raise ValueError('axistypes is not a sequence type')
            except IndexError:
                raise ValueError('more than %d axis types specified' % pyferret.MAX_FERRET_NDIM)
        # axis names
        self._axisnames = [ "" ] * pyferret.MAX_FERRET_NDIM
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
                raise ValueError('axisnames is not a sequence type')
            except IndexError:
                raise ValueError('more than %d axis names specified' % pyferret.MAX_FERRET_NDIM)
        # axis units
        self._axisunits = [ "" ] * pyferret.MAX_FERRET_NDIM
        if axisunits:
            try:
                for k in xrange(len(axisunits)):
                    if self._axistypes[k] != pyferret.AXISTYPE_NORMAL:
                        axunit = axisunits[k]
                        if axunit:
                            if not isinstance(axunit, str): 
                                raise ValueError('axis unit %s is not valid' % str(axtype))
                            self._axisunits[k] = axunit
            except TypeError:
                raise ValueError('axisunits is not a sequence type')
            except IndexError:
                raise ValueError('more than %d axis units specified' % pyferret.MAX_FERRET_NDIM)
        # axis coordinates
	self._axiscoords = [ None ] * pyferret.MAX_FERRET_NDIM
        if axiscoords:
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
                raise ValueError('axiscoords is not a sequence type')
            except IndexError:
                raise ValueError('more than %d axis coordinate arrays specified' % pyferret.MAX_FERRET_NDIM)


