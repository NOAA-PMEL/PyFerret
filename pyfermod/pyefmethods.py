"""
Methods in pyferret to assist in the writing of Ferret external functions 
written in Python.
"""

from pyferret import libpyferret

def ferret_pyfunc():
    """
    A dummy function (which just returns this help message) used to document the
    requirements of python modules used as Ferret external functions (using the
    Ferret command: DEFINE PYFUNC [/NAME=<alias>] <module.name>).  Two methods,
    ferret_init and ferret_compute, must be provided by such a module:


    ferret_init(efid)
        Arguments:
            efid - Ferret's integer ID of this external function

        Returns a dictionary defining the following keys:
            "numargs":      number of input arguments [1 - 9; required]
            "descript":     string description of the function [required]
            "restype":      one of FLOAT_ARRAY or STRING_ARRAY, indicating whether
                            the result is an array of floating-point values or strings
                            [optional, default FLOAT_ARRAY]
            "resstrlen":    if the result type is an array of strings, this specifies
                            the (maximum) length of the strings in the array
                            [optional, default: 128]
            "axes":         6-tuple (X,Y,Z,T,E,F) of result grid axis defining values,
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
            "piecemeal":    6-tuple (X,Y,Z,T,E,F) of True or False indicating if it is
                            acceptable to break up the calculation, if needed, along the
                            corresponding axis [optional; default: False for each axis]
            "argnames":     N-tuple of names for the input arguments
                            [optional; default: (A, B, ...)]
            "argdescripts": N-tuple of descriptions for the input arguments
                            [optional; default: no descriptions]
            "argtypes":     N-tuple of FLOAT_ARRAY, FLOAT_ONEVAL, STRING_ARRAY, or
                            STRING_ONEVAL, indicating whether the input argument is
                            an array of floating-point values, a single floating point
                            value, an array of strings, or a single string value.
                            [optional; default: FLOAT_ARRAY for every argument]
            "influences":   N-tuple of 6-tuples of booleans indicating whether the
                            corresponding input argument's (X,Y,Z,T,E,F) axis influences
                            the result grid's (X,Y,Z,T,E,F) axis.  [optional; default,
                            and when None is given for a 6-tuple: True for every axis]
                      NOTE: If the "influences" value for an axis is True (which is the
                            default), the "axes" value for this axis must be either
                            AXIS_IMPLIED_BY_ARGS (the default) or AXIS_REDUCED.
            "extends":      N-tuple of 6-tuples of pairs of integers.  The n-th tuple,
                            if not None, gives the (X,Y,Z,T,E,F) extension pairs for the
                            n-th argument.  An extension pair, if not None, is the
                            number of points extended in the (low,high) indices for
                            that axis of that argument beyond the implied axis of the
                            result grid.  Thus,
                                    (None, (None, None, None, (-1,1)), None, None, None)
                            means the T axis of the second argument is extended by two
                            points (low dimension lowered by 1, high dimension raised
                            by 1) beyond the implied axis of the result.
                            [optional; default: no extensions assigned]
                      NOTE: If an "extends" pair is given for an axis, the "axes"
                            value for this axis must be AXIS_IMPLIED_BY_ARGS
                            (the default).  The "extends" pair more precisely means
                            the axis in the argument, exactly as provided in the
                            Ferret command, is larger by the indicated amount from
                            the implied result grid axis.

        If an exception is raised, Ferret is notified that an error occurred using
        the message of the exception.


    ferret_compute(efid, result_array, result_bdf, input_arrays, input_bdfs)
        Arguments:
            efid         - Ferret's integer ID of this external function
            result_array - a writeable NumPy float64 ndarray of six dimensions (X,Y,Z,T,E,F)
                           to contain the results of this computation.  The shape and
                           strides of this array has been configured so that only (and
                           all) the data points that should be assigned are accessible.
            result_bdf   - a NumPy ndarray of one dimension containing the bad-data-flag
                           value for the result array.
            input_arrays - tuple of read-only NumPy float64 ndarrays of six dimensions
                           (X,Y,Z,T,E,F) containing the given input data.  The shape and
                           strides of these array have been configured so that only (and
                           all) the data points that should be accessible are accessible.
            input_bdfs   - a NumPy ndarray of one dimension containing
                           the bad-data-flag values for each of the input arrays.

        Any return value is ignored.

        If an exception is raised, Ferret is notified that an error occurred using
        the message of the exception.


    If the dictionary returned from ferret_init assigned a result axis as AXIS_ABSTRACT,
    then the ferret_result_limits method must also be defined:


    ferret_result_limits(efid)
        Arguments:
            efid - Ferret's integer ID of this external function

        Returns a (X,Y,Z,T,E,F) 6-tuple of either None or (low, high) pairs of integers.
        If an axis was not designated as AXIS_ABSTRACT, None should be given for that axis.
        If an axis was designated as AXIS_ABSTRACT, a (low, high) pair of integers should
        be given, and are used as the low and high Ferret indices for that axis.
        [The indices of the NumPy ndarray to be assigned will be from 0 until (high-low)].

        If an exception is raised, Ferret is notified that an error occurred using
        the message of the exception.


    If the dictionary returned from ferret_init assigned a result axis as AXIS_CUSTOM,
    then the ferret_custom_axes method must also be defined:


    ferret_custom_axes(efid)
        Arguments:
            efid - Ferret's integer ID of this external function

        Returns a (X,Y,Z,T,E,F) 6-tuple of either None or a (low, high, delta, unit_name,
        is_modulo) tuple.  If an axis was not designated as AXIS_CUSTOM, None should be
        given for that axis.  If an axis was designated as AXIS_CUSTOM, a (low, high,
        delta, unit_name, is_modulo) tuple should be given where low and high are the
        "world" coordinates (floating point) limits for the axis, delta is the step
        increments in "world" coordinates, unit_name is a string used in describing the
        "world" coordinates, and is_modulo is either True or False, indicating if this
        is a modulo ("periodic" or "wrapping") coordinate system.

        If an exception is raised, Ferret is notified that an error occurred using
        the message of the exception.

    """
    return ferret_pyfunc.__doc__


def get_axis_coordinates(efid, arg, axis):
    """
    Returns the "world" coordinates for an axis of an argument to an external function

    Arguments:
        efid: the Ferret id of the external function
        arg: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9)
        axis: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS,
                                                          T_AXIS, E_AXIS, F_AXIS)
    Returns:
        a NumPy float64 ndarray containing the "world" coordinates,
        or None if the values cannot be determined at the time this was called
    Raises:
        ValueError if efid, arg, or axis is invalid
    """
    # check the efid
    try:
        int_id = int(efid)
        if int_id < 0:
            raise ValueError
    except:
        raise ValueError("efid must be a positive integer value")
    # check the arg index
    try:
        int_arg = int(arg)
        if (int_arg < libpyferret.ARG1) or (int_arg > libpyferret.ARG9):
            raise ValueError
    except:
        raise ValueError("arg must be an integer value in [%d,%d]" % (libpyferret.ARG1,libpyferret.ARG9))
    # check the axis index
    try:
        int_axis = int(axis)
        if (int_axis < libpyferret.X_AXIS) or (int_axis > libpyferret.F_AXIS):
            raise ValueError
    except:
        raise ValueError("axis must be an integer value in [%d,%d]" % (libpyferret.X_AXIS,libpyferret.F_AXIS))
    # make the actual call
    return libpyferret._get_axis_coordinates(int_id, int_arg, int_axis)


def get_axis_box_sizes(efid, arg, axis):
    """
    Returns the "box sizes", in "world" coordinate units,
    for an axis of an argument to an external function

    Arguments:
        efid: the Ferret id of the external function
        arg: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9)
        axis: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS,
                                                          T_AXIS, E_AXIS, F_AXIS)
    Returns:
        a NumPy float64 ndarray containing the "box sizes",
        or None if the values cannot be determined at the time this was called
    Raises:
        ValueError if efid, arg, or axis is invalid
    """
    # check the efid
    try:
        int_id = int(efid)
        if int_id < 0:
            raise ValueError
    except:
        raise ValueError("efid must be a positive integer value")
    # check the arg index
    try:
        int_arg = int(arg)
        if (int_arg < libpyferret.ARG1) or (int_arg > libpyferret.ARG9):
            raise ValueError
    except:
        raise ValueError("arg must be an integer value in [%d,%d]" % (libpyferret.ARG1,libpyferret.ARG9))
    # check the axis index
    try:
        int_axis = int(axis)
        if (int_axis < libpyferret.X_AXIS) or (int_axis > libpyferret.F_AXIS):
            raise ValueError
    except:
        raise ValueError("axis must be an integer value in [%d,%d]" % (libpyferret.X_AXIS,libpyferret.F_AXIS))
    # make the actual call
    return libpyferret._get_axis_box_sizes(int_id, int_arg, int_axis)


def get_axis_box_limits(efid, arg, axis):
    """
    Returns the "box limits", in "world" coordinate units,
    for an axis of an argument to an external function

    Arguments:
        efid: the Ferret id of the external function
        arg: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9)
        axis: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS,
                                                          T_AXIS, E_AXIS, F_AXIS)
    Returns:
        a tuple of two NumPy float64 ndarrays containing the low and high "box limits",
        or None if the values cannot be determined at the time this was called
    Raises:
        ValueError if efid, arg, or axis is invalid
    """
    # check the efid
    try:
        int_id = int(efid)
        if int_id < 0:
            raise ValueError
    except:
        raise ValueError("efid must be a positive integer value")
    # check the arg index
    try:
        int_arg = int(arg)
        if (int_arg < libpyferret.ARG1) or (int_arg > libpyferret.ARG9):
            raise ValueError
    except:
        raise ValueError("arg must be an integer value in [%d,%d]" % (libpyferret.ARG1,libpyferret.ARG9))
    # check the axis index
    try:
        int_axis = int(axis)
        if (int_axis < libpyferret.X_AXIS) or (int_axis > libpyferret.F_AXIS):
            raise ValueError
    except:
        raise ValueError("axis must be an integer value in [%d,%d]" % (libpyferret.X_AXIS,libpyferret.F_AXIS))
    # make the actual call
    return libpyferret._get_axis_box_limits(int_id, int_arg, int_axis)


def get_axis_info(efid, arg, axis):
    """
    Returns information about the axis of an argument to an external function

    Arguments:
        efid: the Ferret id of the external function
        arg: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9)
        axis: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS,
                                                          T_AXIS, E_AXIS, F_AXIS)
    Returns:
        a dictionary defining the following keys:
            "name": name string for the axis coordinate
            "unit": name string for the axis unit
            "backwards": boolean - reversed axis?
            "modulo": float - modulo length of axis, or 0.0 if not modulo
            "regular": boolean - evenly spaced axis?
            "size": number of coordinates on this axis, or -1 if the value
                    cannot be determined at the time this was called
    Raises:
        ValueError if efid, arg, or axis is invalid
    """
    # check the efid
    try:
        int_id = int(efid)
        if int_id < 0:
            raise ValueError
    except:
        raise ValueError("efid must be a positive integer value")
    # check the arg index
    try:
        int_arg = int(arg)
        if (int_arg < libpyferret.ARG1) or (int_arg > libpyferret.ARG9):
            raise ValueError
    except:
        raise ValueError("arg must be an integer value in [%d,%d]" % (libpyferret.ARG1,libpyferret.ARG9))
    # check the axis index
    try:
        int_axis = int(axis)
        if (int_axis < libpyferret.X_AXIS) or (int_axis > libpyferret.F_AXIS):
            raise ValueError
    except:
        raise ValueError("axis must be an integer value in [%d,%d]" % (libpyferret.X_AXIS,libpyferret.F_AXIS))
    # make the actual call
    return libpyferret._get_axis_info(int_id, int_arg, int_axis)


def get_arg_one_val(efid, arg):
    """
    Returns the value of the indicated FLOAT_ONEVAL or STRING_ONEVAL argument.
    Can be called from the ferret_result_limits or ferret_custom_axes method
    of an external function.

    Arguments:
        efid: the Ferret id of the external function
        arg: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9)
    Returns:
        the value of the argument, either as a float (if a FLOAT_ONEVAL)
        or a string (if STRING_ONEVAL)
    Raises:
        ValueError if efid or arg is invalid, or if the argument type is not
        FLOAT_ONEVAL or STRING_ONEVAL
    """
    # check the efid
    try:
        int_id = int(efid)
        if int_id < 0:
            raise ValueError
    except:
        raise ValueError("efid must be a positive integer value")
    # check the arg index
    try:
        int_arg = int(arg)
        if (int_arg < libpyferret.ARG1) or (int_arg > libpyferret.ARG9):
            raise ValueError
    except:
        raise ValueError("arg must be an integer value in [%d,%d]" % (libpyferret.ARG1,libpyferret.ARG9))
    # make the actual call
    return libpyferret._get_arg_one_val(int_id, int_arg)

