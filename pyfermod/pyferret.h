#ifndef PYFERRET_H_
#define PYFERRET_H_

#include <wchar.h>
#include <stdlib.h>

extern float *memory;
extern float *ppl_memory;

#define MAX_FERRET_NDIM 4

/* Enumerated type to assist in creating cdms2.axis objects */
typedef enum AXISTYPE_ {
    AXISTYPE_LONGITUDE = 1,
    AXISTYPE_LATITUDE = 2,
    AXISTYPE_LEVEL = 3,
    AXISTYPE_TIME = 4,
    AXISTYPE_CUSTOM = 5,
    AXISTYPE_ABSTRACT = 6,
    AXISTYPE_NORMAL = 7,
} AXISTYPE;

/* Indices of a time integer array to assist in create cdtime objects */
typedef enum TIMEARRAY_INDEX_ {
    TIMEARRAY_DAYINDEX  = 0,
    TIMEARRAY_MONTHINDEX = 1,
    TIMEARRAY_YEARINDEX = 2,
    TIMEARRAY_HOURINDEX = 3,
    TIMEARRAY_MINUTEINDEX = 4,
    TIMEARRAY_SECONDINDEX = 5,
} TIMEARRAY_INDEX;

/* Enumerated type to assist in creating time cdms2.axis objects */
typedef enum CALTYPE_ {
    CALTYPE_NONE = -1,
    CALTYPE_360DAY = 0,
    CALTYPE_NOLEAP = 50000,
    CALTYPE_GREGORIAN = 52425,
    CALTYPE_JULIAN = 52500,
    CALTYPE_ALLLEAP = 60000,
} CALTYPE;

/* Prototypes for library C functions */
void set_fer_memory(float *mem, size_t mem_size);
void set_ppl_memory(float *mem, int mem_size);
void set_shared_buffer(void);
void decref_pyobj_(void *pyobj_ptr_ptr);
void copy_pystat_data_(float dest[], void *data_ndarray_ptr);

/* Prototypes for library Fortan functions accessed from C routines */
void add_pystat_var_(void *data_ndarray_ptr_ptr, char codename[], char title[], char units[],
                     float *bdfval, char dset[], int axis_nums[MAX_FERRET_NDIM],
                     int axis_starts[MAX_FERRET_NDIM], int axis_ends[MAX_FERRET_NDIM],
                     char errmsg[], int *lenerrmsg, int len_codename, int len_title,
                     int len_units, int len_dset, int maxlen_errmsg);
void clear_fer_last_error_info_(void);
void ef_get_single_axis_info_(int *id, int *argnum, int *axisnum,
                              char axisname[], char axisunit[],
                              int *backwards_axis, int *modulo_axis, int *regular_axis,
                              int maxlen_axisname, int maxlen_axisunit);
void finalize_(void);
void get_axis_num_(int *axisnum, int *axisstart, int *axisend, char axisname[], char axisunit[], 
                   double axiscoords[], int *numcoords, AXISTYPE *axistype, char *errmsg, 
                   int *lenerrmsg, int maxlen_axisname, int maxlen_axisunit, int maxlen_errmsg);
void get_data_array_params_(char dataname[], int *lendataname, float *memory, int *arraystart,
                            int memlo[MAX_FERRET_NDIM], int memhi[MAX_FERRET_NDIM],
                            int steplo[MAX_FERRET_NDIM], int stephi[MAX_FERRET_NDIM],
                            int incr[MAX_FERRET_NDIM], char dataunit[], int *lendataunit,
                            AXISTYPE axtypes[MAX_FERRET_NDIM], float *badval, char errmsg[],
                            int *lenerrmsg, int maxlen_dataname, int maxlen_dataunit, int maxlen_errmsg);
void get_data_array_coordinates_(double axiscoords[], char axisunit[], char axisname[],
                                 int *axisnum, int *numcoords, char errmsg[], int *lenerrmsg,
                                 int maxlen_axisunit, int maxlen_axisname, int maxlen_errmsg);
void get_data_array_time_coords_(int timecoords[][6], CALTYPE *caltype, char axisname[],
                                 int *axisnum, int *numcoords, char errmsg[], int *lenerrmsg,
                                 int maxlen_axisname, int maxlen_errmsg);
void get_fer_last_error_info_(int *errval, char errmsg[], int maxlen_errmsg);
void get_ferret_params_(char errnames[][32], int errvals[], int *numvals);
void get_time_axis_num_(int *axisnum, int *axisstart, int *axisend, char axisname[], 
                        CALTYPE *calendartype, int axiscoords[][6], int *numcoords,
                        char *errmsg, int *lenerrmsg, int maxlen_axisname, int maxlen_errmsg);
void init_journal_(int *status);
void initialize_(void);
void no_journal_(void);
void proclaim_c_(int *ttoutLun, char *leader);
void set_one_cmnd_mode_(int *one_cmnd_mode_int);
void turnoff_verify_(int *status);

/* Functions for Python-backed external functions */

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
