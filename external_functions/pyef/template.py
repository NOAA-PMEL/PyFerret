'''
Template for creating a PyFerret Python External Function (PyEF).
The names of the functions provided should not be changed.  By 
default, PyFerret uses the name of the module as the function name.

Copy this file using a name that you would like to be the function
name, then modify the contents of these functions and comments as
desired. 

@author: 
'''
import numpy


def ferret_init(efid):
    '''
    Initialization function for this PyFerret PyEF.  Returns
    a dictionary describing the features of this PyFerret PyEF.
    At a minimum, assigns the number of arguments expected and 
    a descriptions of the functions.  May also provide 
    descriptions of the arguments and specifications for a 
    non-standard result grid.
    '''
    init_dict = { }

    init_dict["numargs"] = 1
    init_dict["descript"] = "Pass through"

    return init_dict


def ferret_result_limits(efid):
    '''
    Defines the index limits for all abstract axes in the result grid.
    Returns an (X,Y,Z,T,E,F)-axis six-tuple of either (low,high) pairs, 
    for an abstract axis, or None, for a non-abstract axis.  The low 
    and high values are integer values.  If the result grid has no 
    abstract axes, this function will not be called and can be deleted.
    '''
    axis_limits = (None, None, None, None, None, None)

    return axis_limits


def ferret_custom_axes(efid):
    '''
    Defines all custom axes in ther result grid.  Returns a (X,Y,Z,T,E,F)-
    axis six-tuple of either a (low, high, delta, unit_name, is_modulo) 
    tuple, for a custom axis, or None, for a non-custom axis.  The low,
    high, and delta values are floating-point values in units of the axis 
    coordinate ("world coordinates").  If the result grid has no custom
    axes, this function will not be called and can be deleted.
    '''
    axis_info = (None, None, None, None, None, None)
    
    return axis_info


def ferret_compute(efid, result, result_bdf, inputs, input_bdfs):
    '''
    Computation function for this PyFerret PyEF.  Assign values to the
    elements of result; do not reassign result itself.  In other words,
    assign values using notation similar to 'result[...] = ...'; do not
    use notation similar to 'result = ...' as this will simply define
    a new local variable called result, hiding the variable passed into
    this function.

    If an error is detected, raise an appropriate exception.  ValueError
    is commonly used for unexpected values.  IndexError is commonly used 
    for unexpected array sizes.

    Arguments:
        result     - numpy float array to be assigned
        result_bdf - numpy read-only float array of one element giving the 
                     missing-data value to be used when assigning result
        inputs     - tuple of numpy read-only float arrays giving the input
                     values provided by the caller
        input_bdfs - numpy read-only float arrays of one element giving the
                     missing-data value for the corresponding inputs array
    '''
    # Create masks of values that are undefined and that are defined
    bad_mask = ( inputs[0] == input_bdfs[0] )
    good_mask = numpy.logical_not(bad_mask)
    result[good_mask] = inputs[0][good_mask]
    result[bad_mask] = result_bdf
    return

