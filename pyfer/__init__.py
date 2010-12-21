"""
A Python module for running Ferret.

A Python extension module that interfaces with Ferret functionality.
In order to use this module:
    start must first be called to allocate the required memory and
            initialize Ferret
    resize can be used to resize Ferret's allocated memory block
    run is used to submit individual Ferret commands or enter into
            Ferret's command prompting mode
    get is used to retrieve (a copy of) Ferret numeric data arrays
            as NumPy ndarrays
    stop can be used to shutdown Ferret and free the allocated memory.

The FERR_* values are the possible values of err_int in the return
values from the run command.  The err_int return value FERR_OK
indicates no errors.
"""

import sys
import numpy as np

try:
    import cdms2
    import cdtime
except ImportError:
    print >>sys.stderr, "    WARNING: Unable to import cdms2 and/or cdtime; pyferret.get and pyferret.put will fail"

from _pyferret import *

_MAX_FERRET_NDIM = 4

def start(memsize=25.6, journal=True, verify=True):
    """
    Initializes Ferret.  This allocates the initial amount of memory for Ferret
    (from Python-managed memory), opens the journal file, if requested, and sets
    Ferret's verify mode.  This does NOT run any user initialization scripts.

    Arguments:
        memsize: the size, in megawords (where a "word" is 32 bits),
                 to allocate for Ferret's memory block
        journal: turn on Ferret's journal mode?
        verify: turn on Ferret's verify mode?
    Returns:
        True is successful
        False if Ferret has already been started
    Raises:
        ValueError if memsize if not a positive number
        MemoryError if unable to allocate the needed memory
        IOError if unable to open the journal file
    """
    try:
        flt_memsize = float(memsize)
        if flt_memsize <= 0.0:
            raise ValueError
    except:
        raise ValueError, "memsize must be a positive number"

    return _pyferret._start(flt_memsize, bool(journal), bool(verify))


def resize(memsize):
    """
    Resets the the amount of memory allocated for Ferret from Python-managed memory.

    Arguments:
        memsize: the new size, in megawords (where a "word" is 32 bits),
                 for Ferret's memory block
    Returns:
        True if successful - Ferret has the new amount of memory
        False if unsuccessful - Ferret has the previous amount of memory
    Raises:
        ValueError if memsize if not a positive number
        MemoryError if Ferret has not been started or has been stopped
    """
    try:
        flt_memsize = float(memsize)
        if flt_memsize <= 0.0:
            raise ValueError
    except:
        raise ValueError, "memsize must be a positive number"

    return _pyferret._resize(flt_memsize)


def run(command=None):
    """
    Runs a Ferret command just as if entering a command at the Ferret prompt.

    If the command is not given, is None, or is a blank string, Ferret will
    prompt you for commands until "EXIT /TOPYTHON" is given.  In this case,
    the return tuple will be for the last error, if any, that occurred in
    the sequence of commands given to Ferret.

    Arguments:
        command: the Ferret command to be executed.
    Returns:
        (err_int, err_string)
            err_int: one of the FERR_* data values (FERR_OK if there are no errors)
            err_string: error or warning message (can be empty)
        Error messages normally start with "**ERROR"
        Warning messages normally start with "*** NOTE:"
    Raises:
        ValueError if command is neither None nor a String
        MemoryError if Ferret has not been started or has been stopped
    """
    if command == None:
        str_command = ""
    elif not isinstance(command, str):
        raise ValueError, "command must either be None or a string"
    elif command.isspace():
        str_command = ""
    else:
        str_command = command
    retval = _pyferret._run(str_command)
    if (retval[0] == _pyferret._FERR_EXIT_PROGRAM) and (retval[1] == "EXIT"):
        # python -i -c ... intercepts the sys.exit(0) and stays in python
        # so _pyferret._run() makes a C exit(0) call instead and doesn't return
        # This was kept here in case it can be made to work.
        stop()
        sys.exit(0)
    return retval


def get(name, create_mask=True):
    """
    Returns the numeric data array described in name.

    Arguments:
        name: the name of the numeric data array to retrieve
    Returns:
        A cdms2 TransientVariable object (cdms2.tvariable) containing the
        numeric data.  The data, axes, and missing value will be assigned.
        If create_mask is True (or not given), the mask attribute will be
        assigned using the missing value.
    Raises:
        ValueError if the data name is invalid
        MemoryError if Ferret has not been started or has been stopped
    """
    # lists of units (in uppercase) for checking if a custom axis is actual a longitude axis
    UC_LONGITUDE_UNITS = [ "DEG E", "DEG_E", "DEG EAST", "DEG_EAST",
                           "DEGREES E", "DEGREES_E", "DEGREES EAST", "DEGREES_EAST",
                           "DEG W", "DEG_W", "DEG WEST", "DEG_WEST",
                           "DEGREES W", "DEGREES_W", "DEGREES WEST", "DEGREES_WEST" ]
    # lists of units (in uppercase) for checking if a custom axis is actual a latitude axis
    UC_LATITUDE_UNITS  = [ "DEG N", "DEG_N", "DEG NORTH", "DEG_NORTH",
                           "DEGREES N", "DEGREES_N", "DEGREES NORTH", "DEGREES_NORTH",
                           "DEG S", "DEG_S", "DEG SOUTH", "DEG_SOUTH",
                           "DEGREES S", "DEGREES_S", "DEGREES SOUTH", "DEGREES_SOUTH" ]
    # lists of units (in lowercase) for checking if a custom axis can be represented by a cdtime.reltime
    # the unit must be followed by "since" and something else
    LC_TIME_UNITS = [ "s", "sec", "secs", "second", "seconds",
                      "mn", "min", "mins", "minute", "minutes",
                      "hr", "hour", "hours",
                      "dy", "day", "days",
                      "mo", "month", "months",
                      "season", "seasons",
                      "yr", "year", "years" ]
    lc_month_nums = { "jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12 }
    if not isinstance(name, str):
        raise ValueError, "name must be a string"
    elif name.isspace():
        raise ValueError, "name cannot be an empty string"
    # get the data and related information from Ferret
    vals = _pyferret._get(name)
    # break apart the tuple to simplify (returning a dictionary would have been better)
    data = vals[0]
    bdfs = vals[1]
    data_unit = vals[2]
    axis_types = vals[3]
    axis_names = vals[4]
    axis_units = vals[5]
    axis_coords = vals[6]
    # create the axis list for this variable
    var_axes = [ ]
    for k in xrange(_MAX_FERRET_NDIM):
        if axis_types[k] == _pyferret._AXISTYPE_LONGITUDE:
            newaxis = cdms2.createAxis(axis_coords[k],id=axis_names[k])
            newaxis.units = axis_units[k]
            newaxis.designateLongitude()
            var_axes.append(newaxis)
        elif axis_types[k] == _pyferret._AXISTYPE_LATITUDE:
            newaxis = cdms2.createAxis(axis_coords[k],id=axis_names[k])
            newaxis.units = axis_units[k]
            newaxis.designateLatitude()
            var_axes.append(newaxis)
        elif axis_types[k] == _pyferret._AXISTYPE_LEVEL:
            newaxis = cdms2.createAxis(axis_coords[k],id=axis_names[k])
            newaxis.units = axis_units[k]
            newaxis.designateLevel()
            var_axes.append(newaxis)
        elif axis_types[k] == _pyferret._AXISTYPE_TIME:
            # create the time axis from cdtime.comptime (component time) objects
            time_coords = axis_coords[k]
            timevals = [ ]
            for t in xrange(time_coords.shape[0]):
                day = time_coords[t,_pyferret._TIMEARRAY_DAYINDEX]
                month = time_coords[t,_pyferret._TIMEARRAY_MONTHINDEX]
                year = time_coords[t,_pyferret._TIMEARRAY_YEARINDEX]
                hour = time_coords[t,_pyferret._TIMEARRAY_HOURINDEX]
                minute = time_coords[t,_pyferret._TIMEARRAY_MINUTEINDEX]
                second = time_coords[t,_pyferret._TIMEARRAY_SECONDINDEX]
                timevals.append( cdtime.comptime(year,month,day,hour,minute,second) )
            newaxis = cdms2.createAxis(timevals,id=axis_names[k])
            # designate the calendar
            if axis_units[k] == "CALTYPE_360DAY":
                calendar_type = cdtime.Calendar360
            elif axis_units[k] == "CALTYPE_NOLEAP":
                calendar_type = cdtime.NoLeapCalendar
            elif axis_units[k] == "CALTYPE_GREGORIAN":
                calendar_type = cdtime.GregorianCalendar
            elif axis_units[k] == "CALTYPE_JULIAN":
                calendar_type = cdtime.JulianCalendar
            else:
                if axis_units[k] == "CALTYPE_ALLLEAP":
                    raise ValueError, "The all-leap calendar not support by cdms2"
                if axis_units[k] == "CALTYPE_NONE":
                    raise ValueError, "Undesignated calendar not support by cdms2"
                raise RuntimeError, "Unexpected calendar type of %s" % axis_units[k]
            newaxis.designateTime(calendar=calendar_type)
            # and finally append it to the axis list
            var_axes.append(newaxis)
        elif axis_types[k] == _pyferret._AXISTYPE_CUSTOM:
            # custom axis which could be standard axis that is
            # not in Ferret's expected order, so check the units
            uc_units = axis_units[k].upper()
            lc_vals = axis_units[k].lower().split()
            if uc_units in UC_LONGITUDE_UNITS:
                newaxis = cdms2.createAxis(axis_coords[k],id=axis_names[k])
                newaxis.units = axis_units[k]
                newaxis.designateLongitude()
            elif uc_units in UC_LATITUDE_UNITS:
                newaxis = cdms2.createAxis(axis_coords[k],id=axis_names[k])
                newaxis.units = axis_units[k]
                newaxis.designateLatitude()
            elif (len(lc_vals) > 2) and (lc_vals[1] == "since") and (lc_vals[0] in LC_TIME_UNITS):
                # (unit) since (start_date)
                datevals = lc_vals[2].split("-")
                try:
                    # try to convert dd-mon-yyyy ferret-style start_date to yyyy-mm-dd
                    day_num = int(datevals[0])
                    mon_num = lc_month_nums[datevals[1]]
                    year_num = int(datevals[2])
                    lc_vals[2] = "%04d-%02d-%02d" % (year_num, mon_num, day_num)
                    relunit = " ".join(lc_vals)
                except (IndexError, KeyError, ValueError):
                    # use the relative time unit as given
                    relunit = " ".join(lc_vals)
                timevals = [ ]
                for t in xrange(axis_coords[k].shape[0]):
                    dtval = cdtime.reltime(axis_coords[k][t], relunit);
                    timevals.append(dtval)
                newaxis = cdms2.createAxis(timevals,id=axis_names[k])
                newaxis.designateTime()
            else:
                newaxis = cdms2.createAxis(axis_coords[k],id=axis_names[k])
                newaxis.units = axis_units[k]
            var_axes.append(newaxis)
        elif axis_types[k] == _pyferret._AXISTYPE_ABSTRACT:
            newaxis = cdms2.createAxis(axis_coords[k],id=axis_names[k])
            var_axes.append(newaxis)
        elif axis_types[k] == _pyferret._AXISTYPE_NORMAL:
            var_axes.append(None)
        else:
            raise RuntimeError, "Unexpected axis type of %d" % axis_types[k]
    # _pyferret._get returns a copy of the data, so createVariable does not need to make a copy
    datavar = cdms2.createVariable(data, fill_value=bdfs[0], axes=var_axes, id=name,
                                   attributes={"name":name, "units":data_unit})
    if create_mask:
        if np.isnan(bdfs[0]):
            # NaN comparisons always return False, even to another NaN
            datavar.mask = np.isnan(data)
        else:
            # since values in data and bdfs[0] are all float32 values assigned by Ferret,
            # using equality should work correctly
            datavar.mask = ( data == bdfs[0] )
    return datavar


def put(datavar, axis_pos=None, dset='', name=None, title=None):
    """
    Creates a Ferret data variable with a copy of the data given in datavar.

    Arguments:
        datavar:  a cdms2 AbstractVariable describing the data variable
                  to be created in Ferret.  Any masked values in the data
                  of datavar will be set to the missing value for datavar
                  before extracting the data as 32-bit floating-point
                  values for Ferret.
        axis_pos: a four-tuple given the Ferret positions for each axis in
                  datavar.  If the axes in datavar are in (time, level, lat.,
                  long.) order, the tuple (T_AXIS, Z_AXIS, Y_AXIS, X_AXIS)
                  should be used for proper axis handling in Ferret.  If not
                  given (or None), the first longitude axis will be made the
                  X_AXIS, the first latitude axis will be made the Y_AXIS,
                  the first level axis will be made the Z_AXIS, the first
                  time axis will be made the T_AXIS, and any remaining axes
                  are then filled into the remaining unassigned positions.
        dset:     the dataset name or number to be associated with this new
                  variable.  If blank or not given, the current dataset is
                  used.  If None or 'None', no dataset will be associated
                  with the new variable.
        name:     the code name for the variable in Ferret (eg, 'SST').
                  If not given (or None), the value of datavar.id will be
                  used.  The code name must be a non-empty string and
                  (ideally) should not contain whitespace characters.
        title:    the title name for the variable in Ferret (eg, 'Sea
                  Surface Temperature').  If not given (or None), the
                  value of datavar.name, if present, or the value of
                  datavar.id is used.
    Returns:
        None
    Raises:
        AttributeError: if datavar does not have the needed methods or
                        attributes of an AbstractVariable
        MemoryError:    if Ferret has not been started or has been stopped
        ValueError:     if there is a problem with the contents of the arguments
    """
    #
    # code name for the Ferret variable
    if name != None:
        codename = str(name)
    else:
        codename = str(datavar.id)
    if codename.strip() == "":
        raise ValueError, "The code name (id or datavar.id) must be a non-blank string"
    #
    # title name for the Ferret variable
    if title != None:
        titlename = str(title)
    else:
        try:
            titlename = str(datavar.name)
        except AttributeError:
            titlename = codename
    #
    # units for the data
    try:
        data_unit = str(datavar.units)
    except AttributeError:
        data_unit = ""
    #
    # dataset for the variable
    dset_str = str(dset).strip()
    #
    # get the list of axes and initialize the axis information lists
    axis_list = datavar.getAxisList()
    if len(axis_list) > _MAX_FERRET_NDIM:
        raise ValueError, "More than %d axes is not supported in Ferret at this time" % _MAX_FERRET_NDIM
    axis_types = [ _pyferret._AXISTYPE_NORMAL ] * _MAX_FERRET_NDIM
    axis_names = [ "" ] * _MAX_FERRET_NDIM
    axis_units = [ "" ] * _MAX_FERRET_NDIM
    axis_coords = [ None ] * _MAX_FERRET_NDIM
    for k in xrange(len(axis_list)):
        #
        # get the information for this axis
        axis = axis_list[k]
        axis_names[k] = axis.id.strip()
        axis_data = axis.getData()
        try:
            axis_units[k] = axis.units.strip()
        except AttributeError:
            axis_units[k] = ""
        #
        # assign the axis information
        if axis.isLongitude():
            axis_types[k] = _pyferret._AXISTYPE_LONGITUDE
            if not axis_units[k]:
                axis_units[k] = "DEGREES_E"
            axis_coords[k] = np.array(axis_data, dtype=np.float64, copy=1)
        elif axis.isLatitude():
            axis_types[k] = _pyferret._AXISTYPE_LATITUDE
            if not axis_units[k]:
                axis_units[k] = "DEGREES_N"
            axis_coords[k] = np.array(axis_data, dtype=np.float64, copy=1)
        elif axis.isLevel():
            axis_types[k] = _pyferret._AXISTYPE_LEVEL
            axis_coords[k] = np.array(axis_data, dtype=np.float64, copy=1)
        elif axis.isTime():
            #
            # try to create a time axis reading the values as cdtime comptime objects
            try:
                time_coords = np.empty((len(axis_data),6), dtype=np.int32, order="C")
                for t in xrange(len(axis_data)):
                    tval = axis_data[t]
                    time_coords[t,_pyferret._TIMEARRAY_DAYINDEX] = tval.day
                    time_coords[t,_pyferret._TIMEARRAY_MONTHINDEX] = tval.month
                    time_coords[t,_pyferret._TIMEARRAY_YEARINDEX] = tval.year
                    time_coords[t,_pyferret._TIMEARRAY_HOURINDEX] = tval.hour
                    time_coords[t,_pyferret._TIMEARRAY_MINUTEINDEX] = tval.minute
                    time_coords[t,_pyferret._TIMEARRAY_SECONDINDEX] = int(tval.second)
                axis_types[k] = _pyferret._AXISTYPE_TIME
                axis_coords[k] = time_coords
                # assign the axis_units value to the CALTYPE_ calendar type string
                calendar_type = axis.getCalendar()
                if calendar_type == cdtime.Calendar360:
                    axis_units[k] = "CALTYPE_360DAY"
                elif calendar_type == cdtime.NoLeapCalendar:
                    axis_units[k] = "CALTYPE_NOLEAP"
                elif calendar_type == cdtime.GregorianCalendar:
                    axis_units[k] = "CALTYPE_GREGORIAN"
                elif calendar_type == cdtime.JulianCalendar:
                    axis_units[k] = "CALTYPE_JULIAN"
                else:
                    if calendar_type == cdtime.MixedCalendar:
                        raise ValueError, "The cdtime.MixedCalendar not support by pyferret"
                    raise ValueError, "Unknown cdtime calendar %s" % str(calendar_type)
            except AttributeError:
                axis_types[k] = _pyferret._AXISTYPE_CUSTOM
            #
            # if not comptime objects, assume reltime objects - create as a custom axis
            if axis_types[k] == _pyferret._AXISTYPE_CUSTOM:
                time_coords = np.empty((len(axis_data),), dtype=np.float64)
                for t in xrange(len(axis_data)):
                    time_coords[t] = axis_data[t].value
                axis_coords[k] = timecoords
                # assign axis_units as the reltime units - makes sure all are the same
                axis_units[k] = axis_data[0].units
                for t in xrange(1,len(axis_data)):
                    if axis_data[t].units != axis_units[k]:
                        raise ValueError, "Relative time axis does not have a consistent start point"
        #
        # cdms2 will create an axis if None (normal axis) was given, so create a
        # custom or abstract axis only if it does not look like a cdms2-created axis
        elif not ( (axis_units[k] == "") and (len(axis_data) == 1) and (axis_data[0] == 0.0) and \
                   (axis_data.dtype == np.dtype('float64')) and \
                   axis_names[k].startswith("axis_") and axis_names[k][5:].isdigit() ):
            axis_types[k] = _pyferret._AXISTYPE_CUSTOM
            axis_coords[k] = np.array(axis_data, dtype=np.float64, copy=1)
            # if a unitless integer value axis, it is abstract instead of custom
            if axis_units[k] == "":
                axis_int_vals = np.array(axis_data, dtype=int)
                if np.allclose(axis_data, axis_int_vals):
                    axis_types[k] = _pyferret._AXISTYPE_ABSTRACT
    #
    # figure out the desired axis order
    if axis_pos != None:
        # start with the positions provided by the user
        if len(axis_pos) != len(axis_list):
            raise ValueError, "axis_pos, if not None, must give the Ferret positions for each of the axes in datavar"
        ferr_axis = list(axis_pos)
        # append the undefined axes positions, which were initialized to _AXISTYPE_NORMAL
        if not X_AXIS in ferr_axis:
            ferr_axis.append(X_AXIS)
        if not Y_AXIS in ferr_axis:
            ferr_axis.append(Y_AXIS)
        if not Z_AXIS in ferr_axis:
            ferr_axis.append(Z_AXIS)
        if not T_AXIS in ferr_axis:
            ferr_axis.append(T_AXIS)
        # intentionally left as 4 (instead of _MAX_FERRET_NDIM) since added axis will need to be appended
        if len(ferr_axis) != 4:
            raise ValueError, "axis_pos can contain at most one of each of the pyferret integer values X_AXIS, Y_AXIS, Z_AXIS, or T_AXIS"
    else:
        ferr_axis = [ -1 ] * _MAX_FERRET_NDIM
        # assign positions of longitude/latitude/level/time
        for k in xrange(len(axis_types)):
           if axis_types[k] == _pyferret._AXISTYPE_LONGITUDE:
               if not X_AXIS in ferr_axis:
                   ferr_axis[k] = X_AXIS
           elif axis_types[k] == _pyferret._AXISTYPE_LATITUDE:
               if not Y_AXIS in ferr_axis:
                   ferr_axis[k] = Y_AXIS
           elif axis_types[k] == _pyferret._AXISTYPE_LEVEL:
               if not Z_AXIS in ferr_axis:
                   ferr_axis[k] = Z_AXIS
           elif axis_types[k] == _pyferret._AXISTYPE_TIME:
               if not T_AXIS in ferr_axis:
                   ferr_axis[k] = T_AXIS
        # fill in other axes types in unused positions
        if not X_AXIS in ferr_axis:
           ferr_axis[ferr_axis.index(-1)] = X_AXIS
        if not Y_AXIS in ferr_axis:
           ferr_axis[ferr_axis.index(-1)] = Y_AXIS
        if not Z_AXIS in ferr_axis:
           ferr_axis[ferr_axis.index(-1)] = Z_AXIS
        if not T_AXIS in ferr_axis:
           ferr_axis[ferr_axis.index(-1)] = T_AXIS
        try:
           ferr_axis.index(-1)
           raise RuntimeError, "Unexpected undefined axis position (_MAX_FERRET_NDIM increased?) in ferr_axis " + str(ferr_axis)
        except ValueError:
           # expected result
           pass
    #
    # make sure the masked values are set to the missing value
    missingval = datavar.getMissing()
    if np.any(datavar.mask):
        datavar.data[datavar.mask] = missingval
    #
    # get the bad-data-flag value as a 32-bit float
    bdfval = np.array(missingval, dtype=np.float32)
    #
    # get the data as an ndarray of _MAX_FERRET_NDIM dimensions
    # adding new axes still reference the original data array - just creates new shape and stride objects
    data = datavar.data
    for k in xrange(len(axis_list), _MAX_FERRET_NDIM):
        data = data[..., np.newaxis]
    #
    # swap data axes and axis information to give (X_AXIS, Y_AXIS, Z_AXIS, T_AXIS) axes
    # swapping axes still reference the original data array - just creates new shape and stride objects
    k = ferr_axis.index(X_AXIS)
    if k != 0:
        data = data.swapaxes(0, k)
        ferr_axis[0], ferr_axis[k] = ferr_axis[k], ferr_axis[0]
        axis_types[0], axis_types[k] = axis_types[k], axis_types[0]
        axis_names[0], axis_names[k] = axis_names[k], axis_names[0]
        axis_units[0], axis_units[k] = axis_units[k], axis_units[0]
        axis_coords[0], axis_coords[k] = axis_coords[k], axis_coords[0]
    k = ferr_axis.index(Y_AXIS)
    if k != 1:
        data = data.swapaxes(1, k)
        ferr_axis[1], ferr_axis[k] = ferr_axis[k], ferr_axis[1]
        axis_types[1], axis_types[k] = axis_types[k], axis_types[1]
        axis_names[1], axis_names[k] = axis_names[k], axis_names[1]
        axis_units[1], axis_units[k] = axis_units[k], axis_units[1]
        axis_coords[1], axis_coords[k] = axis_coords[k], axis_coords[1]
    k = ferr_axis.index(Z_AXIS)
    if k != 2:
        data = data.swapaxes(2, k)
        ferr_axis[2], ferr_axis[k] = ferr_axis[k], ferr_axis[2]
        axis_types[2], axis_types[k] = axis_types[k], axis_types[2]
        axis_names[2], axis_names[k] = axis_names[k], axis_names[2]
        axis_units[2], axis_units[k] = axis_units[k], axis_units[2]
        axis_coords[2], axis_coords[k] = axis_coords[k], axis_coords[2]
    # T_AXIS must now be ferr_axis[3]
    # assumes _MAX_FERRET_NDIM == 4; extend the logic if axes are added
    # would rather not assume X_AXIS == 0, Y_AXIS == 1, Z_AXIS == 2, T_AXIS == 3
    #
    # now make a copy of the data as (contiguous) 32-bit floats in Fortran order
    fdata = np.array(data, dtype=np.float32, order='F', copy=1)
    #
    # _pyferret._put will throw an Exception if there is a problem
    _pyferret._put(codename, titlename, fdata, bdfval, data_unit, dset_str,
                  axis_types, axis_names, axis_units, axis_coords)
    return None


def stop():
    """
    Shuts down and release all memory used by Ferret.
    After calling this function do not call any Ferret functions except start,
    which will restart Ferret and re-enable the other functions.

    Returns:
        False if Ferret has not been started or has already been stopped
        True otherwise
    """
    return _pyferret._stop()


def ferret_pyfunc():
    """
    A dummy function (which just returns this help message) used to document the
    requirements of python modules used as ferret external functions (using the
    ferret command: DEFINE PYFUNC [/NAME=<alias>] <module.name>).  Two methods,
    ferret_init and ferret_compute, must be provided by such a module:


    ferret_init(id)
        Arguments:
            id - ferret's integer ID of this external function

        Returns a dictionary defining the following keys:
            "numargs":      number of input arguments [1 - 9; required]
            "descript":     string description of the function [required]
            "axes":         4-tuple (X,Y,Z,T) of result grid axis defining values,
                            which are:
                                    AXIS_ABSTRACT:        indexed, ferret_result_limits
                                                          called to define the axis,
                                    AXIS_CUSTOM:          ferret_custom_axes called to
                                                          define the axis,
                                    AXIS_DOES_NOT_EXIST:  does not exist in (normal to)
                                                          the results grid,
                                    AXIS_IMPLIED_BY_ARGS: same as the corresponding axis
                                                          in one or more arguments,
                                    AXIS_REDUCED:         reduced to a single point
                            [optional; default: AXIS_IMPLIED_BY_ARGS for each axis]
            "argnames":     N-tuple of names for the input arguments
                            [optional; default: (A, B, ...)]
            "argdescripts": N-tuple of descriptions for the input arguments
                            [optional; default: no descriptions]
            "influences":   N-tuple of 4-tuples of booleans indicating whether the
                            corresponding input argument's (X,Y,Z,T) axis influences
                            the result grid's (X,Y,Z,T) axis.  [optional; default,
                            and when None is given for a 4-tuple: True for every axis]
                      NOTE: If the "influences" value for an axis is True (which is the
                            default), the "axes" value for this axis must be either
                            AXIS_IMPLIED_BY_ARGS (the default) or AXIS_REDUCED.
            "extends":      N-tuple of 4-tuples of pairs of integers.  The n-th tuple,
                            if not None, gives the (X,Y,Z,T) extension pairs for the
                            n-th argument.  An extension pair, if not None, is the
                            number of points extended in the (low,high) indices for
                            that axis of that argument beyond the implied axis of the
                            result grid.  Thus,
                                    (None, (None, None, None, (-1,1)), None)
                            means the T axis of the second argument is extended by two
                            points (low dimension lowered by 1, high dimension raised
                            by 1) beyond the implied axis of the result.
                            [optional; default: no extensions assigned]
                      NOTE: If an "extends" pair is given for an axis, the "axes"
                            value for this axis must be either AXIS_IMPLIED_BY_ARGS
                            (the default).  The "extends" pair more precisely means
                            the axis in the argument, exactly as provided in the
                            ferret command, is larger by the indicated amount from
                            the implied result grid axis.

        If an exception is raised, Ferret is notified that an error occurred using
        the message of the exception.


    ferret_compute(id, result_array, result_bdf, input_arrays, input_bdfs)
        Arguments:
            id           - ferret's integer ID of this external function
            result_array - a writeable NumPy float32 ndarray of four dimensions (X,Y,Z,T)
                           to contain the results of this computation.  The shape and
                           strides of this array has been configured so that only (and
                           all) the data points that should be assigned are accessible.
            result_bdf   - a NumPy ndarray of one dimension containing the bad-data-flag
                           value for the result array.
            input_arrays - tuple of read-only NumPy float32 ndarrays of four dimensions
                           (X,Y,Z,T) containing the given input data.  The shape and
                           strides of these array have been configured so that only (and
                           all) the data points that should be accessible are accessible.
            input_bdfs   - a NumPy ndarray of one dimension containing
                           the bad-data-flag values for each of the input arrays.

        Any return value is ignored.

        If an exception is raised, Ferret is notified that an error occurred using
        the message of the exception.


    If the dictionary returned from ferret_init assigned a result axis as AXIS_ABSTRACT,
    then the ferret_result_limits method must also be defined:


    ferret_result_limits(id)
        Arguments:
            id - ferret's integer ID of this external function

        Returns a (X,Y,Z,T) 4-tuple of either None or (low, high) pairs of integers.
        If an axis was not designated as AXIS_ABSTRACT, None should be given for that axis.
        If an axis was designated as AXIS_ABSTRACT, a (low, high) pair of integers should
        be given, and are used as the low and high Ferret indices for that axis.
        [The indices of the NumPy ndarray to be assigned will be from 0 until (high-low)].

        If an exception is raised, Ferret is notified that an error occurred using
        the message of the exception.


    If the dictionary returned from ferret_init assigned a result axis as AXIS_CUSTOM,
    then the ferret_custom_axes method must also be defined:


    ferret_custom_axes(id)
        Arguments:
            id - ferret's integer ID of this external function

        Returns a (X,Y,Z,T) 4-tuple of either None or a (low, high, delta, unit_name,
        is_modulo) tuple.  If an axis was not designated as AXIS_CUSTOM, None should be
        given for that axis.  If an axis was designated as AXIS_CUSTOM, a (low, high,
        delta, unit_name, is_modulo) tuple should be given where low and high are the
        "world" coordinates (floating point) limits for the axis, delta is the step
        increments in "world" coordinates, unit_name is a string used in describing the
        "world" coordinates, and is_modulo is either True or False, indicating if this
        is a modulo ("wrapping") coordinate system.

        If an exception is raised, Ferret is notified that an error occurred using
        the message of the exception.

    """
    return ferret_pyfunc.__doc__


def get_axis_coordinates(id, arg, axis):
    """
    Returns the "world" coordinates for an axis of an argument to an external function

    Arguments:
        id: the ferret id of the external function
        arg: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9)
        axis: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS, T_AXIS)
    Returns:
        a NumPy float64 ndarray containing the "world" coordinates,
        or None if the values cannot be determined at the time this was called
    Raises:
        ValueError if id, arg, or axis is invalid
    """
    try:
        int_id = int(id)
        if int_id < 0:
            raise ValueError
    except:
        raise ValueError, "id must be a positive integer value"

    try:
        int_arg = int(arg)
        if (int_arg < ARG1) or (int_arg > ARG9):
            raise ValueError
    except:
        raise ValueError, "arg must be an integer value in [%d,%d]" % (ARG1,ARG9)

    try:
        int_axis = int(axis)
        if (int_axis < X_AXIS) or (int_axis > T_AXIS):
            raise ValueError
    except:
        raise ValueError, "axis must be an integer value in [%d,%d]" % (X_AXIS,T_AXIS)

    return _pyferret._get_axis_coordinates(int_id, int_arg, int_axis)


def get_axis_box_sizes(id, arg, axis):
    """
    Returns the "box sizes", in "world" coordinate units,
    for an axis of an argument to an external function

    Arguments:
        id: the ferret id of the external function
        arg: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9)
        axis: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS, T_AXIS)
    Returns:
        a NumPy float32 ndarray containing the "box sizes",
        or None if the values cannot be determined at the time this was called
    Raises:
        ValueError if id, arg, or axis is invalid
    """
    try:
        int_id = int(id)
        if int_id < 0:
            raise ValueError
    except:
        raise ValueError, "id must be a positive integer value"

    try:
        int_arg = int(arg)
        if (int_arg < ARG1) or (int_arg > ARG9):
            raise ValueError
    except:
        raise ValueError, "arg must be an integer value in [%d,%d]" % (ARG1,ARG9)

    try:
        int_axis = int(axis)
        if (int_axis < X_AXIS) or (int_axis > T_AXIS):
            raise ValueError
    except:
        raise ValueError, "axis must be an integer value in [%d,%d]" % (X_AXIS,T_AXIS)

    return _pyferret._get_axis_box_sizes(int_id, int_arg, int_axis)


def get_axis_box_limits(id, arg, axis):
    """
    Returns the "box limits", in "world" coordinate units,
    for an axis of an argument to an external function

    Arguments:
        id: the ferret id of the external function
        arg: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9)
        axis: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS, T_AXIS)
    Returns:
        a tuple of two NumPy float64 ndarrays containing the low and high "box limits",
        or None if the values cannot be determined at the time this was called
    Raises:
        ValueError if id, arg, or axis is invalid
    """
    try:
        int_id = int(id)
        if int_id < 0:
            raise ValueError
    except:
        raise ValueError, "id must be a positive integer value"

    try:
        int_arg = int(arg)
        if (int_arg < ARG1) or (int_arg > ARG9):
            raise ValueError
    except:
        raise ValueError, "arg must be an integer value in [%d,%d]" % (ARG1,ARG9)

    try:
        int_axis = int(axis)
        if (int_axis < X_AXIS) or (int_axis > T_AXIS):
            raise ValueError
    except:
        raise ValueError, "axis must be an integer value in [%d,%d]" % (X_AXIS,T_AXIS)

    return _pyferret._get_axis_box_limits(int_id, int_arg, int_axis)


def get_axis_info(id, arg, axis):
    """
    Returns information about the axis of an argument to an external function

    Arguments:
        id: the ferret id of the external function
        arg: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9)
        axis: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS, T_AXIS)
    Returns:
        a dictionary defining the following keys:
            "name": name string for the axis coordinate
            "unit": name string for the axis unit
            "backwards": boolean - reversed axis?
            "modulo": boolean - periodic/wrapping axis?
            "regular": boolean - evenly spaced axis?
            "size": number of coordinates on this axis, or -1 if the value
                    cannot be determined at the time this was called
    Raises:
        ValueError if id, arg, or axis is invalid
    """
    try:
        int_id = int(id)
        if int_id < 0:
            raise ValueError
    except:
        raise ValueError, "id must be a positive integer value"

    try:
        int_arg = int(arg)
        if (int_arg < ARG1) or (int_arg > ARG9):
            raise ValueError
    except:
        raise ValueError, "arg must be an integer value in [%d,%d]" % (ARG1,ARG9)

    try:
        int_axis = int(axis)
        if (int_axis < X_AXIS) or (int_axis > T_AXIS):
            raise ValueError
    except:
        raise ValueError, "axis must be an integer value in [%d,%d]" % (X_AXIS,T_AXIS)

    return _pyferret._get_axis_info(int_id, int_arg, int_axis)

