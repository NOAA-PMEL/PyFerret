"""
Methods in pyferret for transferring data between the Ferret engine 
and Python.
"""

from __future__ import print_function

import numpy
import numpy.ma
import io

import libpyferret

# set of units (in uppercase) for checking if a custom axis is actual a longitude axis
_UC_LONGITUDE_UNITS = frozenset( ("DEG E", "DEG_E", "DEG EAST", "DEG_EAST",
                                 "DEGREES E", "DEGREES_E", "DEGREES EAST", "DEGREES_EAST",
                                 "DEG W", "DEG_W", "DEG WEST", "DEG_WEST",
                                 "DEGREES W", "DEGREES_W", "DEGREES WEST", "DEGREES_WEST") )
# set of units (in uppercase) for checking if a custom axis is actual a latitude axis
_UC_LATITUDE_UNITS  = frozenset( ("DEG N", "DEG_N", "DEG NORTH", "DEG_NORTH",
                                  "DEGREES N", "DEGREES_N", "DEGREES NORTH", "DEGREES_NORTH",
                                  "DEG S", "DEG_S", "DEG SOUTH", "DEG_SOUTH",
                                  "DEGREES S", "DEGREES_S", "DEGREES SOUTH", "DEGREES_SOUTH") )

# set of units (in lowercase) for checking if a custom axis can be represented by a cdtime.reltime
# the unit must be followed by "since" and something else
_LC_TIME_UNITS = frozenset( ("s", "sec", "secs", "second", "seconds",
                             "mn", "min", "mins", "minute", "minutes",
                             "hr", "hour", "hours",
                             "dy", "day", "days",
                             "mo", "month", "months",
                             "season", "seasons",
                             "yr", "year", "years") )

_LC_MONTH_NUMS = { "jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6,
                   "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12 }
_UC_MONTH_NAMES = { 1: "JAN", 2:"FEB", 3:"MAR", 4:"APR", 5:"MAY", 6:"JUN",
                    7:"JUL", 8:"AUG", 9:"SEP", 10:"OCT", 11:"NOV", 12:"DEC" }

def metastr(datadict):
    """
    Creates a string representation of the metadata in a data dictionary.
    Print this string to show a nicely formatted display of the metadata.

    Arguments:
        datadict: a data dictionary, as returned by the getdata method.
    Returns:
        the string representation of the metadata in datadict.
    Raises:
        TypeError if datadict is not a dictionary
    """
    if not isinstance(datadict, dict):
        raise TypeError("datadict is not a dictionary")
    # specify an order of output for standard keys, leaving out "data"
    keylist = [ "name", "title", "dset", "data_unit", "missing_value",
                "axis_names", "axis_types", "axis_units", "axis_coords" ]
    # append non-standard keys in alphabetical order
    for key in sorted(datadict.keys()):
        if (key != "data") and (key not in keylist):
            keylist.append(key)
    # create the metadata string using StringIO
    strbuf = io.StringIO()
    for key in keylist:
        try:
            # make sure the key:value pair exists
            val = datadict[key]
            # just in case the key is not a string (for printing)
            keystr = str(key)
            if keystr == "axis_coords":
                print(keystr + ":", file=strbuf)
                for (idx, item) in enumerate(val):
                    # add the axis name (which will be present if coordinates
                    # are given) as a label for the axis coordinates
                    itemlabel = "   '" + datadict["axis_names"][idx] + "': "
                    if datadict["axis_types"][idx] == libpyferret.AXISTYPE_TIME:
                        # add a translation of each of the time 6-tuples
                        strlist = [ ]
                        for subitem in item:
                            strlist.append(" %s = %02d-%3s-%04d %02d:%02d:%02d" % \
                                           (str(subitem),
                                                subitem[libpyferret.TIMEARRAY_DAYINDEX],
                                _UC_MONTH_NAMES[subitem[libpyferret.TIMEARRAY_MONTHINDEX]],
                                                subitem[libpyferret.TIMEARRAY_YEARINDEX],
                                                subitem[libpyferret.TIMEARRAY_HOURINDEX],
                                                subitem[libpyferret.TIMEARRAY_MINUTEINDEX],
                                                subitem[libpyferret.TIMEARRAY_SECONDINDEX],) )
                        if len(strlist) == 0:
                           strlist.append("[]")
                        else:
                           strlist[0] = "[" + strlist[0][1:]
                           strlist[-1] = strlist[-1] + "]"
                    else:
                        # just print the values of non-time axis coordinates
                        strlist = str(item).split('\n')
                    # adjust the subsequent-line-indent if multiple lines
                    itemstr = itemlabel + strlist[0]
                    indent = " " * len(itemlabel)
                    for addstr in strlist[1:]:
                        itemstr += "\n" + indent + addstr
                    print(itemstr, file=strbuf)
            elif keystr == "axis_types":
                # add a translation of the axis type number
                valstr = "("
                for (idx, item) in enumerate(val):
                    if idx > 0:
                        valstr += ", "
                    valstr += str(item)
                    if item == libpyferret.AXISTYPE_LONGITUDE:
                        valstr += "=longitude"
                    elif item == libpyferret.AXISTYPE_LATITUDE:
                        valstr += "=latitude"
                    elif item == libpyferret.AXISTYPE_LEVEL:
                        valstr += "=level"
                    elif item == libpyferret.AXISTYPE_TIME:
                        valstr += "=time"
                    elif item == libpyferret.AXISTYPE_CUSTOM:
                        valstr += "=custom"
                    elif item == libpyferret.AXISTYPE_ABSTRACT:
                        valstr += "=abstract"
                    elif item == libpyferret.AXISTYPE_NORMAL:
                        valstr += "=unused"
                valstr += ")"
                print(keystr + ": " + valstr, file=strbuf)
            elif keystr == "missing_value":
                # print the one value in the missing value array
                print(keystr + ": " + str(val[0]), file=strbuf)
            else:
                # just print as "key: value", except
                # adjust the subsequent-line-indent if multiple lines
                strlist = str(val).split('\n')
                valstr = strlist[0]
                indent = " " * (len(keystr) + 2)
                for addstr in strlist[1:]:
                    valstr += "\n" + indent + addstr
                print(keystr + ": " + valstr, file=strbuf)
        except KeyError:
            # known key not present - ignore
            pass
    strval = strbuf.getvalue()
    strbuf.close()
    return strval


def getstrdata(name):
    """
    Returns the string array and axes information for the data variable
    described in name as a dictionary.

    Arguments:
        name: the name of the string data array to retrieve
    Returns:
        A dictionary contains the string data array and axes information.
        The dictionary contains the following key/value pairs:
            'title' : the string passed in the name argument
            'data': the string data array.  This will be a NumPy String 
                    ndarray with a string length one more than the longest
                    string in the array.
            'missing_value': the missing data value.  This will be a NumPy
                    String ndarray (with the same string length as for data)
                    containing a single String value.
            'axis_types': a list of integer values describing the type of
                    each axis.  Possible values are the following constants
                    defined by the pyferret module:
                        AXISTYPE_LONGITUDE
                        AXISTYPE_LATITUDE
                        AXISTYPE_LEVEL
                        AXISTYPE_TIME
                        AXISTYPE_CUSTOM   (axis units not recognized by Ferret)
                        AXISTYPE_ABSTRACT (axis is unit-less integer values)
                        AXISTYPE_NORMAL   (axis is normal to the data)
            'axis_names': a list of strings giving the name of each axis
            'axis_units': a list of strings giving the unit of each axis.
                    If the axis type is AXISTYPE_TIME, this names the calendar
                    used for the timestamps, as one of the following strings
                    defined by the pyferret module:
                        CALTYPE_360DAY
                        CALTYPE_NOLEAP
                        CALTYPE_GREGORIAN
                        CALTYPE_JULIAN
                        CALTYPE_ALLLEAP
                        CALTYPE_NONE    (calendar not specified)
            'axis_coords': a list of NumPy ndarrays giving the coordinate values
                    for each axis.  If the axis type is neither AXISTYPE_TIME
                    nor AXISTYPE_NORMAL, a NumPy float64 ndarray is given.  If
                    the axis is type AXISTYPE_TIME, a NumPy integer ndarray of
                    shape (N,6) where N is the number of time points.  The six
                    integer values per time point are the day, month, year, hour,
                    minute, and second of the associate calendar for this time
                    axis.  The following constants defined by the pyferret module
                    give the values of these six indices:
                        TIMEARRAY_DAYINDEX
                        TIMEARRAY_MONTHINDEX
                        TIMEARRAY_YEARINDEX
                        TIMEARRAY_HOURINDEX
                        TIMEARRAY_MINUTEINDEX
                        TIMEARRAY_SECONDINDEX
                    (Thus, axis_coords[t, pyferret.TIMEARRAY_YEARINDEX]
                     gives the year of time point t.)
        Note: a relative time axis will be of type AXISTYPE_CUSTOM, with a unit
              indicating the starting point, such as 'days since 01-JAN-2000'
    Raises:
        ValueError if the data name is invalid
        MemoryError if Ferret has not been started or has been stopped
    See also:
        get
    """
    # check name
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    elif name.isspace():
        raise ValueError("name cannot be an empty string")
    # get the data and related information from Ferret
    vals = libpyferret._getstrdata(name)
    # break apart the tuple to simplify (returning a dictionary would have been better)
    data = vals[0]
    bdfs = vals[1]
    axis_types = vals[2]
    axis_names = vals[3]
    axis_units = vals[4]
    axis_coords = vals[5]
    # A custom axis could be standard axis that is not in Ferret's expected order,
    # so check the units
    for k in range(libpyferret.MAX_FERRET_NDIM):
        if axis_types[k] == libpyferret.AXISTYPE_CUSTOM:
            uc_units = axis_units[k].upper()
            if uc_units in _UC_LONGITUDE_UNITS:
                axis_types[k] = libpyferret.AXISTYPE_LONGITUDE
            elif uc_units in _UC_LATITUDE_UNITS:
                axis_types[k] = libpyferret.AXISTYPE_LATITUDE
    # libpyferret._get returns a copy of the data, so no need to force a copy
    return { "title": name, "data":data, "missing_value":bdfs, "axis_types":axis_types, 
             "axis_names":axis_names, "axis_units":axis_units, "axis_coords":axis_coords }


def getdata(name, create_mask=True):
    """
    Returns the numeric array and axes information for the data variable
    described in name as a dictionary.

    Arguments:
        name: the name of the numeric data to retrieve
        create_mask: return the numeric data array as a MaskedArray object?
    Returns:
        A dictionary contains the numeric data array and axes information.
        Note that 'name' is not assigned, which is required for the putdata
        method.  The dictionary contains the following key/value pairs:
            'title' : the string passed in the name argument
            'data': the numeric data array.  If create_mask is True, this
                    will be a NumPy float64 MaskedArray object with the
                    masked array properly assigned.  If create_mask is False,
                    this will just be a NumPy float64 ndarray.
            'missing_value': the missing data value.  This will be a NumPy
                    float64 ndarray containing a single value.
            'data_unit': a string describing the unit of the data.
            'axis_types': a list of integer values describing the type of
                    each axis.  Possible values are the following constants
                    defined by the pyferret module:
                        AXISTYPE_LONGITUDE
                        AXISTYPE_LATITUDE
                        AXISTYPE_LEVEL
                        AXISTYPE_TIME
                        AXISTYPE_CUSTOM   (axis units not recognized by Ferret)
                        AXISTYPE_ABSTRACT (axis is unit-less integer values)
                        AXISTYPE_NORMAL   (axis is normal to the data)
            'axis_names': a list of strings giving the name of each axis
            'axis_units': a list of strings giving the unit of each axis.
                    If the axis type is AXISTYPE_TIME, this names the calendar
                    used for the timestamps, as one of the following strings
                    defined by the pyferret module:
                        CALTYPE_360DAY
                        CALTYPE_NOLEAP
                        CALTYPE_GREGORIAN
                        CALTYPE_JULIAN
                        CALTYPE_ALLLEAP
                        CALTYPE_NONE    (calendar not specified)
            'axis_coords': a list of NumPy ndarrays giving the coordinate values
                    for each axis.  If the axis type is neither AXISTYPE_TIME
                    nor AXISTYPE_NORMAL, a NumPy float64 ndarray is given.  If
                    the axis is type AXISTYPE_TIME, a NumPy integer ndarray of
                    shape (N,6) where N is the number of time points.  The six
                    integer values per time point are the day, month, year, hour,
                    minute, and second of the associate calendar for this time
                    axis.  The following constants defined by the pyferret module
                    give the values of these six indices:
                        TIMEARRAY_DAYINDEX
                        TIMEARRAY_MONTHINDEX
                        TIMEARRAY_YEARINDEX
                        TIMEARRAY_HOURINDEX
                        TIMEARRAY_MINUTEINDEX
                        TIMEARRAY_SECONDINDEX
                    (Thus, axis_coords[t, pyferret.TIMEARRAY_YEARINDEX]
                     gives the year of time point t.)
        Note: a relative time axis will be of type AXISTYPE_CUSTOM, with a unit
              indicating the starting point, such as 'days since 01-JAN-2000'
    Raises:
        ValueError if the data name is invalid
        MemoryError if Ferret has not been started or has been stopped
    See also:
        get
    """
    # check name
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    elif name.isspace():
        raise ValueError("name cannot be an empty string")
    # get the data and related information from Ferret
    vals = libpyferret._get(name)
    # break apart the tuple to simplify (returning a dictionary would have been better)
    data = vals[0]
    bdfs = vals[1]
    data_unit = vals[2]
    axis_types = vals[3]
    axis_names = vals[4]
    axis_units = vals[5]
    axis_coords = vals[6]
    # A custom axis could be standard axis that is not in Ferret's expected order,
    # so check the units
    for k in range(libpyferret.MAX_FERRET_NDIM):
        if axis_types[k] == libpyferret.AXISTYPE_CUSTOM:
            uc_units = axis_units[k].upper()
            if uc_units in _UC_LONGITUDE_UNITS:
                axis_types[k] = libpyferret.AXISTYPE_LONGITUDE
            elif uc_units in _UC_LATITUDE_UNITS:
                axis_types[k] = libpyferret.AXISTYPE_LATITUDE
    # libpyferret._get returns a copy of the data, so no need to force a copy
    if create_mask:
        if numpy.isnan(bdfs[0]):
            # NaN comparisons always return False, even to another NaN
            datavar = numpy.ma.array(data, fill_value=bdfs[0], mask=numpy.isnan(data))
        else:
            # since values in data and bdfs[0] are all float64 values assigned by Ferret,
            # using equality should work correctly
            datavar = numpy.ma.array(data, fill_value=bdfs[0], mask=( data == bdfs[0] ))
    else:
        datavar = data
    return { "title": name, "data":datavar, "missing_value":bdfs, "data_unit":data_unit,
             "axis_types":axis_types, "axis_names":axis_names, "axis_units":axis_units,
             "axis_coords":axis_coords }


def putdata(datavar_dict, axis_pos=None):
    """
    Creates a Ferret data variable with a copy of the data given in the dictionary
    datavar_dict, reordering the data and axes according to tuple axis_pos.

    Arguments:

        datavar_dict: a dictionary with the following keys and associated values:
            'name': the code name for the variable in Ferret (eg, 'SST').
                    Must be given.
            'title': the title name for the variable in Ferret (eg, 'Sea Surface
                    Temperature').  If not given, the value of 'name' is used.
            'dset' : the Ferret dataset name or number to associate with this new data
                    variable.  If blank or not given, the current dataset is used.  If
                    None or 'None', no dataset will be associated with the new variable.
            'data': a NumPy numeric ndarray or masked array.  The data will be saved
                    in Ferret as a 64-bit floating-point values.  Must be given.
            'missing_value': the missing data value.  This will be saved in Ferret as
                    a 64-bit floating-point value.  If not given, Ferret's default
                    missing value (-1.0E34) will be used.
            'data_unit': a string describing the unit of the data.  If not given, no
                    unit will be assigned.
            'axis_types': a list of integer values describing the type of each axis.
                    Possible values are the following constants defined by the pyferret
                    module:
                        AXISTYPE_LONGITUDE
                        AXISTYPE_LATITUDE
                        AXISTYPE_LEVEL
                        AXISTYPE_TIME
                        AXISTYPE_CUSTOM   (axis units not interpreted by Ferret)
                        AXISTYPE_ABSTRACT (axis is unit-less integer values)
                        AXISTYPE_NORMAL   (axis is normal to the data)
                    If not given, AXISTYPE_ABSTRACT will be used if the data array
                    has data for that axis (shape element greater than one); otherwise,
                    AXISTYPE_NORMAL will be used.
            'axis_names': a list of strings giving the name of each axis.  If not given,
                    Ferret will generate names if needed.
            'axis_units': a list of strings giving the unit of each axis.
                    If the axis type is AXISTYPE_TIME, this names the calendar
                    used for the timestamps, as one of the following strings
                    defined by the pyferret module:
                        CALTYPE_360DAY
                        CALTYPE_NOLEAP
                        CALTYPE_GREGORIAN
                        CALTYPE_JULIAN
                        CALTYPE_ALLLEAP
                        CALTYPE_NONE    (calendar not specified)
                    If not given, 'DEGREES_E' will be used for AXISTYPE_LONGITUDE,
                    'DEGREES_N' for AXISTYPE_LATITUDE, CALTYPE_GREGORIAN for
                    AXISTYPE_TIME, and no units will be given for other axis types.
            'axis_coords': a list of arrays of coordinates for each axis.
                    If the axis type is neither AXISTYPE_TIME nor AXISTYPE_NORMAL,
                    a one-dimensional numeric list or ndarray should be given (the
                    values will be stored as floating-point values).
                    If the axis is type AXISTYPE_TIME, a two-dimension list or ndarray
                    with shape (N,6), where N is the number of time points, should be
                    given.  The six integer values per time point are the day, month,
                    year, hour, minute, and second of the associate calendar for this
                    time axis.  The following constants defined by the pyferret module
                    give the values of these six indices:
                        TIMEARRAY_DAYINDEX
                        TIMEARRAY_MONTHINDEX
                        TIMEARRAY_YEARINDEX
                        TIMEARRAY_HOURINDEX
                        TIMEARRAY_MINUTEINDEX
                        TIMEARRAY_SECONDINDEX
                    (Thus, axis_coords[t, pyferret.TIMEARRAY_YEARINDEX] gives the year of
                     time point t.)
                    An array of coordinates must be given if the axis does not have a type
                    of AXISTYPE_NORMAL or AXISTYPE_ABSTRACT (or if axis types are not given).
            Note: a relative time axis should be given as type AXISTYPE_CUSTOM, with a
                  unit indicating the starting point, such as 'days since 01-JAN-2000'

        axis_pos: a six-tuple giving the Ferret positions for each axis in datavar.
            If the axes in datavar are in (forecast, ensemble, time, level, lat., long.)
            order, the tuple (F_AXIS, E_AXIS, T_AXIS, Z_AXIS, Y_AXIS, X_AXIS) should be
            used for proper axis handling in Ferret.  If not given (or None), the first
            longitude axis will be made the X_AXIS, the first latitude axis will be made
            the Y_AXIS, the first level axis will be made the Z_AXIS, the first time axis
            will be made the T_AXIS, the second time axis will be made the F_AXIS, and any
            remaining axes are then filled into the remaining unassigned positions.

    Returns:
        None

    Raises:
        KeyError: if datavar_dict is missing a required key
        MemoryError: if Ferret has not been started or has been stopped
        ValueError:  if there is a problem with the value of a key

    See also:
        put
    """
    #
    # code name for the variable
    codename = datavar_dict.get('name', '')
    if codename != None:
        codename = str(codename).strip()
    if not codename:
        raise ValueError("The value of 'name' must be a non-blank string")
    #
    # title for the variable
    titlename = str(datavar_dict.get('title', codename)).strip()
    #
    # Ferret dataset for the variable; None gets turned into the string 'None'
    dset_str = str(datavar_dict.get('dset', '')).strip()
    #
    # value for missing data
    missingval = float(datavar_dict.get('missing_value', -1.0E34))
    #
    # data units
    data_unit = str(datavar_dict.get('data_unit', '')).strip()
    #
    # axis types
    axis_types = [ libpyferret.AXISTYPE_NORMAL ] * libpyferret.MAX_FERRET_NDIM
    given_axis_types = datavar_dict.get('axis_types', None)
    if given_axis_types:
        if len(given_axis_types) > libpyferret.MAX_FERRET_NDIM:
            raise ValueError("More than %d axes (in the types) is not supported in Ferret at this time" % libpyferret.MAX_FERRET_NDIM)
        for k in range(len(given_axis_types)):
            axis_types[k] = given_axis_types[k]
    #
    # axis names
    axis_names = [ "" ] * libpyferret.MAX_FERRET_NDIM
    given_axis_names = datavar_dict.get('axis_names', None)
    if given_axis_names:
        if len(given_axis_names) > libpyferret.MAX_FERRET_NDIM:
            raise ValueError("More than %d axes (in the names) is not supported in Ferret at this time" % libpyferret.MAX_FERRET_NDIM)
        for k in range(len(given_axis_names)):
            axis_names[k] = given_axis_names[k]
    #
    # axis units
    axis_units = [ "" ] * libpyferret.MAX_FERRET_NDIM
    given_axis_units = datavar_dict.get('axis_units', None)
    if given_axis_units:
        if len(given_axis_units) > libpyferret.MAX_FERRET_NDIM:
            raise ValueError("More than %d axes (in the units) is not supported in Ferret at this time" % libpyferret.MAX_FERRET_NDIM)
        for k in range(len(given_axis_units)):
            axis_units[k] = given_axis_units[k]
    # axis coordinates
    axis_coords = [ None ] * libpyferret.MAX_FERRET_NDIM
    given_axis_coords = datavar_dict.get('axis_coords', None)
    if given_axis_coords:
        if len(given_axis_coords) > libpyferret.MAX_FERRET_NDIM:
            raise ValueError("More than %d axes (in the coordinates) is not supported in Ferret at this time" % libpyferret.MAX_FERRET_NDIM)
        for k in range(len(given_axis_coords)):
            axis_coords[k] = given_axis_coords[k]
    #
    # data array
    datavar = datavar_dict['data']
    #
    # For any axis with data (shape > 1), if AXISTYPE_NORMAL (presumably from not being specified),
    # change to AXISTYPE_ABSTRACT.  Note that a shape == 1 could either be normal or a singleton axis.
    try:
        shape = datavar.shape
        if len(shape) > libpyferret.MAX_FERRET_NDIM:
            raise ValueError("More than %d axes (in the data) is not supported in Ferret at this time" % libpyferret.MAX_FERRET_NDIM)
        for k in range(len(shape)):
            if (shape[k] > 1) and (axis_types[k] == libpyferret.AXISTYPE_NORMAL):
                axis_types[k] = libpyferret.AXISTYPE_ABSTRACT
    except AttributeError:
        raise ValueError("The value of 'data' must be a NumPy ndarray (or derived from an ndarray)")
    #
    # assign any defaults on the axis information not given,
    # and make a copy of the axis coordinates (to ensure they are well-behaved)
    for k in range(libpyferret.MAX_FERRET_NDIM):
        if axis_types[k] == libpyferret.AXISTYPE_LONGITUDE:
            if not axis_units[k]:
                axis_units[k] = "DEGREES_E"
            axis_coords[k] = numpy.array(axis_coords[k], dtype=numpy.float64, copy=1)
            if axis_coords[k].shape[0] != shape[k]:
                raise ValueError("number of coordinates for axis %d does not match the number of data points" % (k+1))
        elif axis_types[k] == libpyferret.AXISTYPE_LATITUDE:
            if not axis_units[k]:
                axis_units[k] = "DEGREES_N"
            axis_coords[k] = numpy.array(axis_coords[k], dtype=numpy.float64, copy=1)
            if axis_coords[k].shape[0] != shape[k]:
                raise ValueError("number of coordinates for axis %d does not match the number of data points" % (k+1))
        elif axis_types[k] == libpyferret.AXISTYPE_LEVEL:
            axis_coords[k] = numpy.array(axis_coords[k], dtype=numpy.float64, copy=1)
            if axis_coords[k].shape[0] != shape[k]:
                raise ValueError("number of coordinates for axis %d does not match the number of data points" % (k+1))
        elif axis_types[k] == libpyferret.AXISTYPE_TIME:
            if not axis_units[k]:
                axis_units[k] = CALTYPE_GREGORIAN
            axis_coords[k] = numpy.array(axis_coords[k], dtype=numpy.int32, order='C', copy=1)
            if axis_coords[k].shape[0] != shape[k]:
                raise ValueError("number of coordinates for axis %d does not match the number of data points" % (k+1))
            if axis_coords[k].shape[1] != 6:
                raise ValueError("number of components (second index) for time axis %d is not 6" % (k+1))
        elif axis_types[k] == libpyferret.AXISTYPE_CUSTOM:
            axis_coords[k] = numpy.array(axis_coords[k], dtype=numpy.float64, copy=1)
            if axis_coords[k].shape[0] != shape[k]:
                raise ValueError("number of coordinates for axis %d does not match the number of data points" % (k+1))
        elif axis_types[k] == libpyferret.AXISTYPE_ABSTRACT:
            if axis_coords[k] != None:
                axis_coords[k] = numpy.array(axis_coords[k], dtype=numpy.float64, copy=1)
                if axis_coords[k].shape[0] != shape[k]:
                    raise ValueError("number of coordinates for axis %d does not match the number of data points" % (k+1))
            else:
                # axis needed but not specified
                axis_coords[k] = numpy.arange(1.0, float(shape[k]) + 0.5, 1.0, dtype=numpy.float64)
        elif axis_types[k] == libpyferret.AXISTYPE_NORMAL:
            axis_coords[k] = None
        else:
            raise RuntimeError("Unexpected axis_type of %d" % axis_types[k])
    #
    # figure out the desired axis order
    if axis_pos != None:
        # start with the positions provided by the user
        ferr_axis = list(axis_pos)
        if len(ferr_axis) < len(shape):
            raise ValueError("axis_pos, if given, must provide a position for each axis in the data")
        # append undefined axes positions, which were initialized to AXISTYPE_NORMAL
        if not libpyferret.X_AXIS in ferr_axis:
            ferr_axis.append(libpyferret.X_AXIS)
        if not libpyferret.Y_AXIS in ferr_axis:
            ferr_axis.append(libpyferret.Y_AXIS)
        if not libpyferret.Z_AXIS in ferr_axis:
            ferr_axis.append(libpyferret.Z_AXIS)
        if not libpyferret.T_AXIS in ferr_axis:
            ferr_axis.append(libpyferret.T_AXIS)
        if not libpyferret.E_AXIS in ferr_axis:
            ferr_axis.append(libpyferret.E_AXIS)
        if not libpyferret.F_AXIS in ferr_axis:
            ferr_axis.append(libpyferret.F_AXIS)
        # intentionally left as 6 (instead of MAX_FERRET_NDIM) since new axes will need to be appended
        if len(ferr_axis) != 6:
            raise ValueError("axis_pos can contain at most one of each of the pyferret integer values X_AXIS, Y_AXIS, Z_AXIS, or T_AXIS")
    else:
        ferr_axis = [ -1 ] * libpyferret.MAX_FERRET_NDIM
        # assign positions of longitude/latitude/level/time
        for k in range(len(axis_types)):
            if axis_types[k] == libpyferret.AXISTYPE_LONGITUDE:
                if not libpyferret.X_AXIS in ferr_axis:
                    ferr_axis[k] = libpyferret.X_AXIS
            elif axis_types[k] == libpyferret.AXISTYPE_LATITUDE:
                if not libpyferret.Y_AXIS in ferr_axis:
                    ferr_axis[k] = libpyferret.Y_AXIS
            elif axis_types[k] == libpyferret.AXISTYPE_LEVEL:
                if not libpyferret.Z_AXIS in ferr_axis:
                    ferr_axis[k] = libpyferret.Z_AXIS
            elif axis_types[k] == libpyferret.AXISTYPE_TIME:
                if not libpyferret.T_AXIS in ferr_axis:
                    ferr_axis[k] = libpyferret.T_AXIS
                elif not libpyferret.F_AXIS in ferr_axis:
                    ferr_axis[k] = libpyferret.F_AXIS
        # fill in other axes types in unused positions
        if not libpyferret.X_AXIS in ferr_axis:
            ferr_axis[ferr_axis.index(-1)] = libpyferret.X_AXIS
        if not libpyferret.Y_AXIS in ferr_axis:
            ferr_axis[ferr_axis.index(-1)] = libpyferret.Y_AXIS
        if not libpyferret.Z_AXIS in ferr_axis:
            ferr_axis[ferr_axis.index(-1)] = libpyferret.Z_AXIS
        if not libpyferret.T_AXIS in ferr_axis:
            ferr_axis[ferr_axis.index(-1)] = libpyferret.T_AXIS
        if not libpyferret.E_AXIS in ferr_axis:
            ferr_axis[ferr_axis.index(-1)] = libpyferret.E_AXIS
        if not libpyferret.F_AXIS in ferr_axis:
            ferr_axis[ferr_axis.index(-1)] = libpyferret.F_AXIS
        try:
            ferr_axis.index(-1)
            raise RuntimeError("Unexpected undefined axis position (MAX_FERRET_NDIM increased?) in ferr_axis " + str(ferr_axis))
        except ValueError:
            # expected result
            pass
    #
    # get the missing data value as a 64-bit float
    bdfval = numpy.array(missingval, dtype=numpy.float64)
    #
    # if a masked array, make sure the masked values are set
    # to the missing value, and get the ndarray underneath
    try:
        if numpy.any(datavar.mask):
            datavar.data[datavar.mask] = bdfval
        data = datavar.data
    except AttributeError:
        data = datavar
    #
    # get the data as an ndarray of MAX_FERRET_NDIM dimensions
    # adding new axes still reference the original data array - just creates new shape and stride objects
    for k in range(len(shape), libpyferret.MAX_FERRET_NDIM):
        data = data[..., numpy.newaxis]
    #
    # swap data axes and axis information to give (X_AXIS, Y_AXIS, Z_AXIS, T_AXIS, E_AXIS, F_AXIS) axes
    # swapping axes still reference the original data array - just creates new shape and stride objects
    k = ferr_axis.index(libpyferret.X_AXIS)
    if k != 0:
        data = data.swapaxes(0, k)
        ferr_axis[0], ferr_axis[k] = ferr_axis[k], ferr_axis[0]
        axis_types[0], axis_types[k] = axis_types[k], axis_types[0]
        axis_names[0], axis_names[k] = axis_names[k], axis_names[0]
        axis_units[0], axis_units[k] = axis_units[k], axis_units[0]
        axis_coords[0], axis_coords[k] = axis_coords[k], axis_coords[0]
    k = ferr_axis.index(libpyferret.Y_AXIS)
    if k != 1:
        data = data.swapaxes(1, k)
        ferr_axis[1], ferr_axis[k] = ferr_axis[k], ferr_axis[1]
        axis_types[1], axis_types[k] = axis_types[k], axis_types[1]
        axis_names[1], axis_names[k] = axis_names[k], axis_names[1]
        axis_units[1], axis_units[k] = axis_units[k], axis_units[1]
        axis_coords[1], axis_coords[k] = axis_coords[k], axis_coords[1]
    k = ferr_axis.index(libpyferret.Z_AXIS)
    if k != 2:
        data = data.swapaxes(2, k)
        ferr_axis[2], ferr_axis[k] = ferr_axis[k], ferr_axis[2]
        axis_types[2], axis_types[k] = axis_types[k], axis_types[2]
        axis_names[2], axis_names[k] = axis_names[k], axis_names[2]
        axis_units[2], axis_units[k] = axis_units[k], axis_units[2]
        axis_coords[2], axis_coords[k] = axis_coords[k], axis_coords[2]
    k = ferr_axis.index(libpyferret.T_AXIS)
    if k != 3:
        data = data.swapaxes(3, k)
        ferr_axis[3], ferr_axis[k] = ferr_axis[k], ferr_axis[3]
        axis_types[3], axis_types[k] = axis_types[k], axis_types[3]
        axis_names[3], axis_names[k] = axis_names[k], axis_names[3]
        axis_units[3], axis_units[k] = axis_units[k], axis_units[3]
        axis_coords[3], axis_coords[k] = axis_coords[k], axis_coords[3]
    k = ferr_axis.index(libpyferret.E_AXIS)
    if k != 4:
        data = data.swapaxes(4, k)
        ferr_axis[4], ferr_axis[k] = ferr_axis[k], ferr_axis[4]
        axis_types[4], axis_types[k] = axis_types[k], axis_types[4]
        axis_names[4], axis_names[k] = axis_names[k], axis_names[4]
        axis_units[4], axis_units[k] = axis_units[k], axis_units[4]
        axis_coords[4], axis_coords[k] = axis_coords[k], axis_coords[4]
    # F_AXIS must now be ferr_axis[5]
    # assumes MAX_FERRET_NDIM == 6; extend the logic if axes are added
    # would rather not assume X_AXIS == 0, Y_AXIS == 1, Z_AXIS == 2,
    #                         T_AXIS == 3, E_AXIS == 4, F_AXIS == 5
    #
    # now make a copy of the data as (contiguous) 64-bit floats in Fortran order
    fdata = numpy.array(data, dtype=numpy.float64, order='F', copy=1)
    #
    # libpyferret._put will raise an Exception if there is a problem
    libpyferret._put(codename, titlename, fdata, bdfval, data_unit, dset_str,
                  axis_types, axis_names, axis_units, axis_coords)
    return None


def get(name, create_mask=True):
    """
    Returns the numeric data array described in name as a TransientVariable object.

    Arguments:
        name: the name of the numeric data array to retrieve
        create_mask: create the mask for the TransientVariable object?
    Returns:
        A cdms2 TransientVariable object (cdms2.tvariable) containing the
        numeric data.  The data, axes, and missing value will be assigned.
        If create_mask is True (or not given), the mask attribute will be
        assigned using the missing value.
    Raises:
        ImportError: if the cdms2 or cdtime modules cannot be found (use getdata instead)
        ValueError:  if the data name is invalid
        MemoryError: if Ferret has not been started or has been stopped
    See also:
        getdata (does not need cdms2 or cdtime)
    """
    # Check (again) if able to import cdms2/cdtime.
    # If the imports were successful before, these imports do nothing
    try:
        import cdms2
        import cdtime
    except ImportError:
        raise ImportError("cdms2 or cdtime not found; pyferret.get not available.\n" \
                          "             Use pyferret.getdata instead.")

    # get the data and related information from Ferret,
    # building on what was done in getdata
    data_dict = getdata(name, create_mask)
    data = data_dict["data"]
    bdfs = data_dict["missing_value"]
    data_unit = data_dict["data_unit"]
    axis_types = data_dict["axis_types"]
    axis_names = data_dict["axis_names"]
    axis_units = data_dict["axis_units"]
    axis_coords = data_dict["axis_coords"]
    # create the axis list for this variable
    var_axes = [ ]
    for k in range(libpyferret.MAX_FERRET_NDIM):
        if axis_types[k] == libpyferret.AXISTYPE_LONGITUDE:
            newaxis = cdms2.createAxis(axis_coords[k], id=axis_names[k])
            newaxis.units = axis_units[k]
            newaxis.designateLongitude()
            var_axes.append(newaxis)
        elif axis_types[k] == libpyferret.AXISTYPE_LATITUDE:
            newaxis = cdms2.createAxis(axis_coords[k], id=axis_names[k])
            newaxis.units = axis_units[k]
            newaxis.designateLatitude()
            var_axes.append(newaxis)
        elif axis_types[k] == libpyferret.AXISTYPE_LEVEL:
            newaxis = cdms2.createAxis(axis_coords[k], id=axis_names[k])
            newaxis.units = axis_units[k]
            newaxis.designateLevel()
            var_axes.append(newaxis)
        elif axis_types[k] == libpyferret.AXISTYPE_TIME:
            # create the time axis from cdtime.comptime (component time) objects
            time_coords = axis_coords[k]
            timevals = [ ]
            for t in range(time_coords.shape[0]):
                day = time_coords[t, libpyferret.TIMEARRAY_DAYINDEX]
                month = time_coords[t, libpyferret.TIMEARRAY_MONTHINDEX]
                year = time_coords[t, libpyferret.TIMEARRAY_YEARINDEX]
                hour = time_coords[t, libpyferret.TIMEARRAY_HOURINDEX]
                minute = time_coords[t, libpyferret.TIMEARRAY_MINUTEINDEX]
                second = time_coords[t, libpyferret.TIMEARRAY_SECONDINDEX]
                timevals.append( cdtime.comptime(year,month,day,hour,minute,second) )
            newaxis = cdms2.createAxis(timevals, id=axis_names[k])
            # designate the calendar
            if axis_units[k] == CALTYPE_360DAY:
                calendar_type = cdtime.Calendar360
            elif axis_units[k] == CALTYPE_NOLEAP:
                calendar_type = cdtime.NoLeapCalendar
            elif axis_units[k] == CALTYPE_GREGORIAN:
                calendar_type = cdtime.GregorianCalendar
            elif axis_units[k] == CALTYPE_JULIAN:
                calendar_type = cdtime.JulianCalendar
            else:
                if axis_units[k] == CALTYPE_ALLLEAP:
                    raise ValueError("The all-leap calendar not support by cdms2")
                if axis_units[k] == CALTYPE_NONE:
                    raise ValueError("Undesignated calendar not support by cdms2")
                raise RuntimeError("Unexpected calendar type of %s" % axis_units[k])
            newaxis.designateTime(calendar=calendar_type)
            # and finally append it to the axis list
            var_axes.append(newaxis)
        elif axis_types[k] == libpyferret.AXISTYPE_CUSTOM:
            # Check a custom axis for relative time units.  Note that getdata has
            # already dealt with longitude or latitude not in Ferret's standard position.
            lc_vals = axis_units[k].lower().split()
            if (len(lc_vals) > 2) and (lc_vals[1] == "since") and (lc_vals[0] in _LC_TIME_UNITS):
                # (unit) since (start_date)
                datevals = lc_vals[2].split("-")
                try:
                    # try to convert dd-mon-yyyy Ferret-style start_date to yyyy-mm-dd
                    day_num = int(datevals[0])
                    mon_num = _LC_MONTH_NUMS[datevals[1]]
                    year_num = int(datevals[2])
                    lc_vals[2] = "%04d-%02d-%02d" % (year_num, mon_num, day_num)
                    relunit = " ".join(lc_vals)
                except (IndexError, KeyError, ValueError):
                    # use the relative time unit as given
                    relunit = " ".join(lc_vals)
                timevals = [ ]
                for t in range(axis_coords[k].shape[0]):
                    dtval = cdtime.reltime(axis_coords[k][t], relunit)
                    timevals.append(dtval)
                newaxis = cdms2.createAxis(timevals, id=axis_names[k])
                newaxis.designateTime()
            else:
                newaxis = cdms2.createAxis(axis_coords[k], id=axis_names[k])
                newaxis.units = axis_units[k]
            var_axes.append(newaxis)
        elif axis_types[k] == libpyferret.AXISTYPE_ABSTRACT:
            newaxis = cdms2.createAxis(axis_coords[k], id=axis_names[k])
            var_axes.append(newaxis)
        elif axis_types[k] == libpyferret.AXISTYPE_NORMAL:
            var_axes.append(None)
        else:
            raise RuntimeError("Unexpected axis type of %d" % axis_types[k])
    # getdata returns a copy of the data, thus createVariable does not
    # need to force a copy.  The mask, if request, was created by getdata.
    datavar = cdms2.createVariable(data, fill_value=bdfs[0], axes=var_axes,
                                   attributes={"name":name, "units":data_unit})
    return datavar


def put(datavar, axis_pos=None):
    """
    Creates a Ferret data variable with a copy of the data given in the
    AbstractVariable object datavar.

    Arguments:

        datavar:  a cdms2 AbstractVariable describing the data variable
                  to be created in Ferret.  Any masked values in the data
                  of datavar will be set to the missing value for datavar
                  before extracting the data as 64-bit floating-point
                  values for Ferret.  In addition to the data and axes
                  described in datavar, the following attributes are used:
                      id: the code name for the variable in Ferret (eg,
                          'SST').  This name must be present and, ideally,
                          should not contain spaces, quotes, or algebraic
                          symbols.
                      name: the title name of the variable in Ferret (eg,
                          'Sea Surface Temperature').  If not present, the
                          value of the id attribute is used.
                      units: the unit name for the data.  If not present,
                          no units are associated with the data.
                      dset: Ferret dataset name or number to be associated
                          with this new variable.  If not given or blank,
                          the variable is associated with the current dataset.
                          If None or 'None', the variable is not associated
                          with any dataset.

        axis_pos: a six-tuple giving the Ferret positions for each axis in
                  datavar.  If the axes in datavar are in (forecast, ensemble,
                  time, level, lat., long.) order, the tuple (F_AXIS, E_AXIS,
                  T_AXIS, Z_AXIS, Y_AXIS, X_AXIS) should be used for proper
                  axis handling in Ferret.  If not given (or None), the first
                  longitude axis will be made the X_AXIS, the first latitude
                  axis will be made the Y_AXIS, the first level axis will be
                  made the Z_AXIS, the first time axis will be made the T_AXIS,
                  the second time axis will be made the F_AXIS, and any remaining
                  axes are then filled into the remaining unassigned positions.

    Returns:
        None

    Raises:
        ImportError:    if the cdms2 or cdtime modules cannot be found (use putdata instead)
        AttributeError: if datavar is missing a required method or attribute
        MemoryError:    if Ferret has not been started or has been stopped
        ValueError:     if there is a problem with the contents of the arguments

    See also:
        putdata (does not require cdms2 or cdtime)
    """
    # Check (again) if able to import cdms2/cdtime.
    # If the imports were successful before, these imports do nothing
    try:
        import cdms2
        import cdtime
    except ImportError:
        raise ImportError("cdms2 or cdtime not found; pyferret.put not available.\n" \
                          "             Use pyferret.putdata instead.")


    #
    # code name for the Ferret variable
    codename = datavar.id.strip()
    if codename == "":
        raise ValueError("The id attribute must be a non-blank string")
    #
    # title name for the Ferret variable
    try:
        titlename = datavar.name.strip()
    except AttributeError:
        titlename = codename
    #
    # units for the data
    try:
        data_unit = datavar.units.strip()
    except AttributeError:
        data_unit = ""
    #
    # missing data value
    missingval = datavar.getMissing()
    #
    # Ferret dataset for the variable; None / 'None' is different from blank / empty
    try:
        dset_str = str(datavar.dset).strip()
    except AttributeError:
        dset_str = ""
    #
    # get the list of axes and initialize the axis information lists
    axis_list = datavar.getAxisList()
    if len(axis_list) > libpyferret.MAX_FERRET_NDIM:
        raise ValueError("More than %d axes is not supported in Ferret at this time" % libpyferret.MAX_FERRET_NDIM)
    axis_types = [ libpyferret.AXISTYPE_NORMAL ] * libpyferret.MAX_FERRET_NDIM
    axis_names = [ "" ] * libpyferret.MAX_FERRET_NDIM
    axis_units = [ "" ] * libpyferret.MAX_FERRET_NDIM
    axis_coords = [ None ] * libpyferret.MAX_FERRET_NDIM
    for k in range(len(axis_list)):
        #
        # get the information for this axis
        axis = axis_list[k]
        axis_names[k] = axis.id.strip()
        try:
            axis_units[k] = axis.units.strip()
        except AttributeError:
            axis_units[k] = ""
        axis_data = axis.getData()
        #
        # assign the axis information
        if axis.isLongitude():
            axis_types[k] = libpyferret.AXISTYPE_LONGITUDE
            axis_coords[k] = axis_data
        elif axis.isLatitude():
            axis_types[k] = libpyferret.AXISTYPE_LATITUDE
            axis_coords[k] = axis_data
        elif axis.isLevel():
            axis_types[k] = libpyferret.AXISTYPE_LEVEL
            axis_coords[k] = axis_data
        elif axis.isTime():
            #
            # try to create a time axis reading the values as cdtime comptime objects
            try:
                time_coords = numpy.empty((len(axis_data),6), dtype=numpy.int32, order="C")
                for t in range(len(axis_data)):
                    tval = axis_data[t]
                    time_coords[t, libpyferret.TIMEARRAY_DAYINDEX] = tval.day
                    time_coords[t, libpyferret.TIMEARRAY_MONTHINDEX] = tval.month
                    time_coords[t, libpyferret.TIMEARRAY_YEARINDEX] = tval.year
                    time_coords[t, libpyferret.TIMEARRAY_HOURINDEX] = tval.hour
                    time_coords[t, libpyferret.TIMEARRAY_MINUTEINDEX] = tval.minute
                    time_coords[t, libpyferret.TIMEARRAY_SECONDINDEX] = int(tval.second)
                axis_types[k] = libpyferret.AXISTYPE_TIME
                axis_coords[k] = time_coords
                # assign the axis_units value to the CALTYPE_ calendar type string
                calendar_type = axis.getCalendar()
                if calendar_type == cdtime.Calendar360:
                    axis_units[k] = CALTYPE_360DAY
                elif calendar_type == cdtime.NoLeapCalendar:
                    axis_units[k] = CALTYPE_NOLEAP
                elif calendar_type == cdtime.GregorianCalendar:
                    axis_units[k] = CALTYPE_GREGORIAN
                elif calendar_type == cdtime.JulianCalendar:
                    axis_units[k] = CALTYPE_JULIAN
                else:
                    if calendar_type == cdtime.MixedCalendar:
                        raise ValueError("The cdtime.MixedCalendar not support by pyferret")
                    raise ValueError("Unknown cdtime calendar %s" % str(calendar_type))
            except AttributeError:
                axis_types[k] = libpyferret.AXISTYPE_CUSTOM
            #
            # if not comptime objects, assume reltime objects - create as a custom axis
            if axis_types[k] == libpyferret.AXISTYPE_CUSTOM:
                time_coords = numpy.empty((len(axis_data),), dtype=numpy.float64)
                for t in range(len(axis_data)):
                    time_coords[t] = axis_data[t].value
                axis_coords[k] = timecoords
                # assign axis_units as the reltime units - makes sure all are the same
                axis_units[k] = axis_data[0].units
                for t in range(1, len(axis_data)):
                    if axis_data[t].units != axis_units[k]:
                        raise ValueError("Relative time axis does not have a consistent start point")
        #
        # cdms2 will create an axis if None (normal axis) was given, so create a
        # custom or abstract axis only if it does not look like a cdms2-created axis
        elif not ( (axis_units[k] == "") and (len(axis_data) == 1) and (axis_data[0] == 0.0) and \
                   (axis_data.dtype == numpy.dtype('float64')) and \
                   axis_names[k].startswith("axis_") and axis_names[k][5:].isdigit() ):
            axis_types[k] = libpyferret.AXISTYPE_CUSTOM
            axis_coords[k] = axis_data
            # if a unitless integer value axis, it is abstract instead of custom
            if axis_units[k] == "":
                axis_int_vals = numpy.array(axis_data, dtype=int)
                if numpy.allclose(axis_data, axis_int_vals):
                    axis_types[k] = libpyferret.AXISTYPE_ABSTRACT
    #
    # datavar is an embelished masked array
    datavar_dict = { 'name': codename, 'title': titlename, 'dset': dset_str, 'data': datavar,
                     'missing_vaue': missingval, 'data_unit': data_unit, 'axis_types': axis_types,
                     'axis_names': axis_names, 'axis_units': axis_units, 'axis_coords': axis_coords }
    #
    # use putdata to set defaults, rearrange axes, and add copies
    # of data in the appropriate format to Ferret
    putdata(datavar_dict, axis_pos)
    return None

