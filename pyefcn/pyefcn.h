/*
 *  This software was developed by the Thermal Modeling and Analysis
 *  Project(TMAP) of the National Oceanographic and Atmospheric
 *  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
 *  hereafter referred to as NOAA/PMEL/TMAP.
 *
 *  Access and use of this software shall impose the following
 *  obligations and understandings on the user. The user is granted the
 *  right, without any fee or cost, to use, copy, modify, alter, enhance
 *  and distribute this software, and any derivative works thereof, and
 *  its supporting documentation for any purpose whatsoever, provided
 *  that this entire notice appears in all copies of the software,
 *  derivative works and supporting documentation.  Further, the user
 *  agrees to credit NOAA/PMEL/TMAP in any publications that result from
 *  the use of this software or in any product that includes this
 *  software. The names TMAP, NOAA and/or PMEL, however, may not be used
 *  in any advertising or publicity to endorse or promote any products
 *  or commercial entity unless specific written permission is obtained
 *  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
 *  is not obligated to provide the user with any support, consulting,
 *  training or assistance of any kind with regard to the use, operation
 *  and performance of this software nor to provide the user with any
 *  updates, revisions, new versions or "bug fixes".
 *
 *  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
 *  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY
 *  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
 *  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
 *  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
 *  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef PYEFCN_H_
#define PYEFCN_H_

#define MAX_FERRET_NDIM 4
#define INIT_METHOD_NAME "ferret_init"
#define COMPUTE_METHOD_NAME "ferret_compute"
#define CUSTOM_AXES_METHOD_NAME "ferret_custom_axes"
#define RESULT_LIMITS_METHOD_NAME "ferret_result_limits"

/*
 * Initialization routine for the ferret python external function associated
 * with the python module "modname".  This function calls the "ferret_init"
 * method in the module to get a dictionary of information in order to assign
 * the result grid information in ferret.  The signature of the ferret_init
 * method should be:
 *
 * ferdict = ferret_init(id)
 *
 * where the argument, id, is ferret's integer ID of this external function
 * and the return value, ferdict, is a dictionary defining:
 *     "numargs": number of input arguments [1 - 9, required]
 *     "descript": string description of the function [required]
 *     "axes": 4-tuple (X,Y,Z,T) of result grid axis defining values (defined in the pyferret
 *             module), which are:
 *                 AXIS_ABSTRACT: indexed, ferret_result_limits called to define the axis,
 *                 AXIS_CUSTOM: ferret_custom_axes called to define the axis,
 *                 AXIS_DOES_NOT_EXIST: does not exist in (normal to) the results grid,
 *                 AXIS_IMPLIED_BY_ARGS: same as the corresponding axis in one or more arguments,
 *                 AXIS_REDUCED: reduced to a single point
 *             [optional, default: AXIS_IMPLIED_BY_ARGS for each axis]
 *     "argnames": N-tuple of names for the input arguments [optional, default: (A, B, ...)]
 *     "argdescripts": N-tuple of descriptions for the input arguments
 *                     [optional, default: no descriptions]
 *     "influences": N-tuple of 4-tuples of booleans indicating whether the corresponding input
 *                   argument's (X,Y,Z,T) axis influences the result grid's (X,Y,Z,T) axis.
 *                   [optional, default and when None is given for a 4-tuple: True for every axis]
 *     "extends": N-tuple of 4-tuples of pairs of integers.  The n-th tuple, if not None, gives
 *                the (X,Y,Z,T) extension pairs for the n-th argument.  An extension pair, if not
 *                None, is the number of points to extend in the (low,high) direction for that axis
 *                of that argument when passed to the ferret_compute function.  Thus,
 *                    (None, (None, None, None, (-1,1)), None)
 *                will expand the T axis of the second argument by 2 points (low dimension lowered
 *                by 1, high dimension raised by 1).  [optional, default: no extensions assigned]
 *
 * If an exception is raised, Ferret is notified that an error occurred using
 * the message of the exception.
 *
 * Arguments for the pyefcn_init function:
 *     id - ferret's id for the function associated with this python module
 *     modname - name of the Python module, null terminated.  This module name should
 *               be in a form appropriate for the python import statement; eg,
 *               "package.module" ('.' to separate module containers; no extension).
 *     errmsg - a character array to contain any error messages generated
 *              which should be null terminated
 *
 * Returns an empty error message if and only if successful.
 */
void pyefcn_init(int id, char modname[], char errmsg[]);


/*
 * Interface function for calling the ferret_result_limits method in the python
 * module "modname".  The values returned by this method are used to assign the
 * limits in ferret of result grid axes that were designated as AXIS_ABSTRACT in
 * the dictionary returned from the call to the ferret_init method in this module.
 * The pyferret method get_axis_coordinates may be useful in determining the values
 * to return.  The signature of the ferret_result_limits method should be:
 *
 * limits_tuple = ferret_result_limits(id)
 *
 * where the argument, id, is ferret's integer ID of this external function and the
 * return value, limits_tuple, is a (X,Y,Z,T) 4-tuple of either None or (low, high)
 * pairs of integers.  If an axis was not designated as AXIS_ABSTRACT, None should
 * be given for that axis.  If an axis was designated as AXIS_ABSTRACT, a (low, high)
 * pair of two integers should be given, which are used as the low and high Ferret
 * indices for that axis.  [The indices of the NumPy ndarray to be assigned will be
 * from 0 until (high-low)].
 *
 * If an exception is raised, Ferret is notified that an error occurred using
 * the message of the exception.
 *
 *
 * Arguments for the pyefcn_result_limits function:
 *     id - ferret's id for the function associated with this python module
 *     modname - name of the Python module, null terminated.  This module name should
 *               be in a form appropriate for the python import statement; eg,
 *               "package.module" ('.' to separate module containers; no extension).
 *     errmsg - a character array to contain any error messages generated
 *              which should be null terminated
 *
 * Returns an empty error message if and only if successful.
 */
void pyefcn_result_limits(int id, char modname[], char errmsg[]);


/*
 * Interface function for calling the ferret_custom_axes method in the python
 * module "modname".  The values returned by this method are used to assign in
 * ferret the data for result grid axes that were designated as AXIS_CUSTOM in
 * the dictionary returned from the call to the ferret_init method in this module.
 * The pyferret method get_axis_coordinates may be useful in determining the values
 * to return.  The signature of the ferret_custom_axes method should be:
 *
 * data_tuple = ferret_custom_axes(id)
 *
 * where the argument, id, is ferret's integer ID of this external function and the
 * return value, data_tuple, is a (X,Y,Z,T) 4-tuple of either None or a (low, high,
 * delta, unit_name, is_modulo) tuple.  If an axis was not designated as AXIS_CUSTOM,
 * None should be given for that axis.  If an axis was designated as AXIS_CUSTOM, a
 * (low, high, delta, unit_name, is_modulo) tuple should be given where low and high
 * are the "world" coordinates (floating point) limits for the axis, delta is the step
 * increments in "world" coordinates, unit_name is a string used in describing the
 * "world" coordinates, and is_modulo is either True or False, indicating if this is
 * a modulo ("wrapping") coordinate system.
 *
 * If an exception is raised, Ferret is notified that an error occurred using
 * the message of the exception.
 *
 *
 * Arguments for the pyefcn_custom_axes function:
 *     id - ferret's id for the function associated with this python module
 *     modname - name of the Python module, null terminated.  This module name should
 *               be in a form appropriate for the python import statement; eg,
 *               "package.module" ('.' to separate module containers; no extension).
 *     errmsg - a character array to contain any error messages generated
 *              which should be null terminated
 *
 * Returns an empty error message if and only if successful.
 */
void fer_pyefcn_custom_axes(int id, char modname[], char errmsg[]);


/*
 * Compute interface function for the ferret python external function
 * associated with the python module "modname".  This function calls the
 * "ferret_compute" method in the module to assign the result data grid.
 * The signature of the ferret_compute method should be:
 *
 *     ferret_compute(id, result_array, result_badval, input_arrays, input_badvals)
 *
 * where:
 *     id - ferret's integer ID of this external function
 *     result_array - a writeable NumPy float32 ndarray of four
 *         dimensions (X,Y,Z,T) to contain the results of this
 *         computation.  The shape and strides of this array has
 *         been configured so that only (and all) the data points
 *         that should be assigned are accessible.
 *     result_badval - a NumPy ndarray of one dimension containing
 *         the bad-data-flag value for the result array.
 *     input_arrays - tuple of read-only NumPy float32 ndarrays of
 *         four dimensions (X,Y,Z,T) containing the given input data.
 *         The shape and strides of these array have been configured
 *         so that only (and all) the data points that should be
 *         accessible are accessible.
 *     input_badvals - a NumPy ndarray of one dimension containing
 *         the bad-data-flag values for each of the input arrays.
 *
 * Any return value from ferret_compute is ignored.
 * If an exception is raised, Ferret is notified that an error occurred using
 * the message of the exception.
 *
 *
 * Arguments for the pyefcn_compute function:
 *
 *     id - ferret's id for the function associated with this python module
 *     modname - name of the Python module, null-terminated.  This module name should
 *               be in a form appropriate for the python import statement; eg,
 *               "parent.module" ('.' to separate module containers; no extension).
 *     arrays - an array of Fortran float grid arrays.  The first array, arrays[0],
 *              is the Fortran array to be assigned with the result grid data.
 *              The remaining arrays are the input grid data Fortran arrays.
 *              The Fortran dimensions of these arrays are:
 *                      arrays[k](memlo[k][0]:memhi[k][0],  - X
 *                                memlo[k][1]:memhi[k][1],  - Y
 *                                memlo[k][2]:memhi[k][2],  - Z
 *                                memlo[k][3]:memhi[k][3])  - T
 *     numarrays - the number of arrays given above (including the results array).
 *     memlo, memhi - Fortran array dimensions as given above.
 *     steplo, stephi, incr - for loop limits and increments for assigning
 *             Fortran data arrays.  Loops start at steplo, increment by incr,
 *             and stop *after* stephi.  The ([k][0], [k][1], [k][2], [k][3])
 *             values are for the (X,Y,Z,T) axes or arrays[k].
 *     badvals - the bad-data-flag value for each of the arrays.
 *     errmsg - a character array to contain any error messages generated,
 *             which should be null-terminated
 *
 * Returns an empty error message if and only if successful.
 */
void pyefcn_compute(int id, char modname[], float *arrays[], int numarrays,
                    int memlo[][MAX_FERRET_NDIM], int memhi[][MAX_FERRET_NDIM],
                    int steplo[][MAX_FERRET_NDIM], int stephi[][MAX_FERRET_NDIM],
                    int incr[][MAX_FERRET_NDIM], float badvals[], char errmsg[]);

/*
 * Returns the message from a Python exception, and clear the exception.
 * If an exception was raised with no message, a default error message
 * ("Exception raised with no message") is returned.  If an exception had
 * not been raised, an empty string is returned.
 */
char *pyefcn_get_error(void);

#endif
