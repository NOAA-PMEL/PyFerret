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

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL pyferret_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "ferret.h"
#include "pyferret.h"
#include "EF_Util.h"

static const char *AXIS_NAMES[MAX_FERRET_NDIM] = { "X", "Y", "Z", "T", "E", "F" };

/*
 * See pyferret.h for information on this function
 */
void pyefcn_init(int id, char modname[], char errmsg[])
{
    PyObject  *valobj;
    PyObject  *usermod;
    PyObject  *initdict;
    int        num_args;
    int        restype;
    int        resstrlen;
    char      *strptr;
    char       descript[EF_MAX_DESCRIPTION_LENGTH];
    PyObject  *seqobj;
    int        seqlen;
    int        j, k, q;
    int        axisvals[MAX_FERRET_NDIM];
    int        axisredu[MAX_FERRET_NDIM];
    int        rsltaxes[MAX_FERRET_NDIM];
    PyObject  *itemobj;
    char       name[EF_MAX_NAME_LENGTH];
    int        argtype;
    PyObject  *subseqobj;
    int        subseqlen;
    PyObject  *subsubseqobj;
    int        subsubseqlen;
    int        deltas[2];
    int        val;
    PyObject  *keysobj;

    /*
     * Import the user's Python module
     */
#if PY_MAJOR_VERSION > 2
    valobj = PyUnicode_FromString(modname);
#else
    valobj = PyString_FromString(modname);
#endif
    if ( valobj == NULL ) {
        PyErr_Clear();
        sprintf(errmsg, "Problems creating a Python string from the module name: %s", modname);
        return;
    }
    usermod = PyImport_Import(valobj);
    /* valobj no longer needed */
    Py_DECREF(valobj);
    /* check for errors */
    if ( usermod == NULL ) {
        PyErr_Clear();
        sprintf(errmsg, "Unable to import module: %s", modname);
        return;
    }

    /*
     * Call the initialization method in the user's python module
     * with the ferret function ID as the sole argument
     */
    initdict = PyObject_CallMethod(usermod, INIT_METHOD_NAME, "i", id);
    /* usermod no longer needed */
    Py_DECREF(usermod);
    /* check for errors */
    if ( initdict == NULL ) {
        sprintf(errmsg, "Error when calling %s in %s: %s", INIT_METHOD_NAME, modname, pyefcn_get_error());
        return;
    }

    /*
     * Process the contents of the dictionary returned
     */
    if ( ! PyDict_Check(initdict) ) {
        Py_DECREF(initdict);
        sprintf(errmsg, "Invalid return value (not a dictionary) from %s in %s",
                        INIT_METHOD_NAME, modname);
        return;
    }

    /*
     * "numargs": number of input arguments [1 - 9, required]
     */
    valobj = PyDict_GetItemString(initdict, "numargs"); /* borrowed reference */
    if ( valobj == NULL ) {
        Py_DECREF(initdict);
        sprintf(errmsg, "\"numargs\" not defined in the dictionary returned from %s in %s",
                        INIT_METHOD_NAME, modname);
        return;
    }
#if PY_MAJOR_VERSION > 2
    num_args = (int) PyLong_AsLong(valobj);
#else
    num_args = (int) PyInt_AsLong(valobj);
#endif
    if ( (num_args < 1) || (num_args > EF_MAX_ARGS) ) {
        PyErr_Clear();
        Py_DECREF(initdict);
        strcpy(errmsg, "Invalid \"numargs\" value (not an integer [0-9])");
        return;
    }
    FORTRAN(ef_set_num_args)(&id, &num_args);

    /*
     * "descript": string description of the function [required]
     */
    valobj = PyDict_GetItemString(initdict, "descript"); /* borrowed reference */
    if ( valobj == NULL ) {
        Py_DECREF(initdict);
        sprintf(errmsg, "\"descript\" not defined in the dictionary returned from %s in %s",
                        INIT_METHOD_NAME, modname);
        return;
    }
#if PY_MAJOR_VERSION > 2
    strptr = PyUnicode_AsUTF8(valobj);
#else
    strptr = PyString_AsString(valobj);
#endif
    if ( strptr == NULL ) {
        PyErr_Clear();
        Py_DECREF(initdict);
        strcpy(errmsg, "Invalid \"descript\" value (not a string)");
        return;
    }
    strncpy(descript, strptr, EF_MAX_DESCRIPTION_LENGTH);
    descript[EF_MAX_DESCRIPTION_LENGTH - 1] = '\0';
    FORTRAN(ef_set_desc_sub)(&id, descript);

    /*
     * "restype": type of the result argument [optional, default: FLOAT_ARRAY]
     */
    valobj = PyDict_GetItemString(initdict, "restype"); /* borrowed reference */
    if ( valobj != NULL ) {
#if PY_MAJOR_VERSION > 2
        restype = (int) PyLong_AsLong(valobj);
#else
        restype = (int) PyInt_AsLong(valobj);
#endif
        if ( restype == FLOAT_ARRAY ) {
            restype = FLOAT_RETURN;
        }
        else if ( restype == STRING_ARRAY ) {
            restype = STRING_RETURN;
        }
        else {
            PyErr_Clear();
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"restype\" value (not FLOAT_ARRAY nor STRING_ARRAY)");
            return;
        }
    }
    else {
        /* Key not present but no error raised */
        restype = FLOAT_RETURN;
    }
    FORTRAN(ef_set_result_type)(&id, &restype);

    /*
     * "resstrlen": (maximum) length of strings in the result string array [optional]
     * Not used here; just check the value if given.  Default assigned by pyefcn_compute
     */
    valobj = PyDict_GetItemString(initdict, "resstrlen"); /* borrowed reference */
    if ( valobj != NULL ) {
#if PY_MAJOR_VERSION > 2
        resstrlen = (int) PyLong_AsLong(valobj);
#else
        resstrlen = (int) PyInt_AsLong(valobj);
#endif
        if ( resstrlen < 1 ) {
            PyErr_Clear();
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"resstrlen\" value (not a positive integer)");
            return;
        }
    }

    /*
     * "axes": 6-tuple (X,Y,Z,T,E,F) of result grid axis defining values:
     *             AXIS_ABSTRACT: indexed, ferret_result_limits called to define the axis,
     *             AXIS_CUSTOM: ferret_custom_axis called to define the axis,
     *             AXIS_DOES_NOT_EXIST: does not exist in (normal to) the results grid,
     *             AXIS_IMPLIED_BY_ARGS: same as the corresponding axis in one or more arguments,
     *             AXIS_REDUCED: reduced to a single point
     *         [optional, default: AXIS_IMPLIED_BY_ARGS for each value]
     */
    valobj = PyDict_GetItemString(initdict, "axes"); /* borrowed reference */
    if ( valobj != NULL ) {
        seqobj = PySequence_Fast(valobj, "axes value");
        if ( seqobj == NULL ) {
            PyErr_Clear();
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"axes\" value (not a tuple or list)");
            return;
        }
        seqlen = (int) PySequence_Fast_GET_SIZE(seqobj);
        if ( seqlen > MAX_FERRET_NDIM ) {
            Py_DECREF(seqobj);
            Py_DECREF(initdict);
            sprintf(errmsg, "Invalid \"axes\" value (tuple or list with more than %d items)",
                             MAX_FERRET_NDIM);
            return;
        }
    }
    else {
        seqobj = NULL;
        seqlen = -1;
    }
    for (k = 0; k < MAX_FERRET_NDIM; k++) {
        axisvals[k] = IMPLIED_BY_ARGS;
        axisredu[k] = RETAINED;
        rsltaxes[k] = IMPLIED_BY_ARGS;
        if ( k < seqlen ) {
            itemobj = PySequence_Fast_GET_ITEM(seqobj, (Py_ssize_t) k); /* borrowed reference */
#if PY_MAJOR_VERSION > 2
            val = (int) PyLong_AsLong(itemobj);
#else
            val = (int) PyInt_AsLong(itemobj);
#endif
            switch( val ) {
                case IMPLIED_BY_ARGS:
                    break;
                case ABSTRACT:
                    axisvals[k] = ABSTRACT;
                    rsltaxes[k] = ABSTRACT;
                    break;
                case CUSTOM:
                    axisvals[k] = CUSTOM;
                    rsltaxes[k] = CUSTOM;
                    break;
                case NORMAL:
                    axisvals[k] = NORMAL;
                    rsltaxes[k] = NORMAL;
                    break;
                case REDUCED:
                    axisredu[k] = REDUCED;
                    rsltaxes[k] = REDUCED;
                    break;
                case -1:
                    PyErr_Clear();
                    /* FALLTHRU */
                default:
                    Py_DECREF(seqobj);
                    Py_DECREF(initdict);
                    sprintf(errmsg, "Invalid \"axes\" value (not one of the AXIS_* values) for the %s axis",
                                    AXIS_NAMES[k]);
                    return;
            }
        }
    }
    Py_XDECREF(seqobj);
    FORTRAN(ef_set_axis_inheritance_6d)(&id, &(axisvals[0]), &(axisvals[1]), &(axisvals[2]), 
                                     &(axisvals[3]), &(axisvals[4]), &(axisvals[5]));
    FORTRAN(ef_set_axis_reduction_6d)(&id, &(axisredu[0]), &(axisredu[1]), &(axisredu[2]), 
                                   &(axisredu[3]), &(axisredu[4]), &(axisredu[5]));

    /*
     * "piecemeal": 6-tuple (X,Y,Z,T,E,F) allow breaking calculations along this result grid axis?
     *         [optional, default: False for each value]
     */
    valobj = PyDict_GetItemString(initdict, "piecemeal"); /* borrowed reference */
    if ( valobj != NULL ) {
        seqobj = PySequence_Fast(valobj, "piecemeal value");
        if ( seqobj == NULL ) {
            PyErr_Clear();
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"piecemeal\" value (not a tuple or list)");
            return;
        }
        seqlen = (int) PySequence_Fast_GET_SIZE(seqobj);
        if ( seqlen > MAX_FERRET_NDIM ) {
            Py_DECREF(seqobj);
            Py_DECREF(initdict);
            sprintf(errmsg, "Invalid \"piecemeal\" value (tuple or list with more than %d items)",
                             MAX_FERRET_NDIM);
            return;
        }
    }
    else {
        seqobj = NULL;
        seqlen = -1;
    }
    for (k = 0; k < MAX_FERRET_NDIM; k++) {
        axisvals[k] = NO;
        if ( k < seqlen ) {
            itemobj = PySequence_Fast_GET_ITEM(seqobj, (Py_ssize_t) k); /* borrowed reference */
            /* Must be one of the singleton objects Py_True or Py_False to be accepted */
            if ( itemobj == Py_True ) {
                axisvals[k] = YES;
            }
            else if ( itemobj != Py_False ) {
                PyErr_Clear();
                Py_DECREF(seqobj);
                Py_DECREF(initdict);
                strcpy(errmsg, "Invalid \"piecemeal\" value (not True or False)");
                return;
            }
        }
    }
    Py_XDECREF(seqobj);
    FORTRAN(ef_set_piecemeal_ok_6d)(&id, &(axisvals[0]), &(axisvals[1]), &(axisvals[2]), 
                                 &(axisvals[3]), &(axisvals[4]), &(axisvals[5]));

    /*
     * "argnames": N-tuple of names for the input arguments [optional, default: (A, B, ...)]
     */
    valobj = PyDict_GetItemString(initdict, "argnames"); /* borrowed reference */
    if ( valobj != NULL ) {
        seqobj = PySequence_Fast(valobj, "argnames value");
        if ( seqobj == NULL ) {
            PyErr_Clear();
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"argnames\" value (not a tuple or list)");
            return;
        }
        seqlen = (int) PySequence_Fast_GET_SIZE(seqobj);
        if ( seqlen > num_args ) {
            Py_DECREF(seqobj);
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"argnames\" value (tuple or list with too many items)");
            return;
        }
    }
    else {
        seqobj = NULL;
        seqlen = -1;
    }
    for (j = 0; j < num_args; j++) {
        /* Assign a default name */
        name[0] = (char) ('A' + j);
        name[1] = '\0';
        if ( j < seqlen ) {
            /* Get the name from the tuple */
            itemobj = PySequence_Fast_GET_ITEM(seqobj, (Py_ssize_t) j); /* borrowed reference */
#if PY_MAJOR_VERSION > 2
            strptr = PyUnicode_AsUTF8(itemobj);
#else
            strptr = PyString_AsString(itemobj);
#endif
            if ( strptr != NULL ) {
                strncpy(name, strptr, EF_MAX_NAME_LENGTH);
                name[EF_MAX_NAME_LENGTH-1] = '\0';
            }
            else {
                Py_DECREF(seqobj);
                Py_DECREF(initdict);
                sprintf(errmsg, "Invalid \"argnames\" value (not a string) for the ARG%d", j+1);
                return;
            }
        }
        q = j+1;
        FORTRAN(ef_set_arg_name_sub)(&id, &q, name);
    }
    Py_XDECREF(seqobj);

    /*
     * "argdescripts": N-tuple of descriptions for the input arguments [optional, default: no descriptions]
     */
    valobj = PyDict_GetItemString(initdict, "argdescripts"); /* borrowed reference */
    if ( valobj != NULL ) {
        seqobj = PySequence_Fast(valobj, "argdescripts value");
        if ( seqobj == NULL ) {
            PyErr_Clear();
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"argdescripts\" value (not a tuple or list)");
            return;
        }
        seqlen = (int) PySequence_Fast_GET_SIZE(seqobj);
        if ( seqlen > num_args ) {
            Py_DECREF(seqobj);
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"argdescripts\" value (tuple or list with too many items)");
            return;
        }
    }
    else {
        seqobj = NULL;
        seqlen = -1;
    }
    for (j = 0; j < num_args; j++) {
        /* Assign the default description */
        descript[0] = '\0';
        if ( j < seqlen ) {
            /* Get the description from the tuple */
            itemobj = PySequence_Fast_GET_ITEM(seqobj, (Py_ssize_t) j); /* borrowed reference */
#if PY_MAJOR_VERSION > 2
            strptr = PyUnicode_AsUTF8(itemobj);
#else
            strptr = PyString_AsString(itemobj);
#endif
            if ( strptr != NULL ) {
                strncpy(descript, strptr, EF_MAX_DESCRIPTION_LENGTH);
                descript[EF_MAX_DESCRIPTION_LENGTH-1] = '\0';
            }
            else {
                PyErr_Clear();
                Py_DECREF(seqobj);
                Py_DECREF(initdict);
                sprintf(errmsg, "Invalid \"argdescripts\" value (not a string) for the ARG%d", j+1);
                return;
            }
        }
        q = j + 1;
        FORTRAN(ef_set_arg_desc_sub)(&id, &q, descript);
    }
    Py_XDECREF(seqobj);

    /*
     * "argtypes": N-tuple of types for the input arguments [optional, default: FLOAT_ARRAY]
     *             FLOAT_ARRAY, FLOAT_ONEVAL, STRING_ARRAY, and STRING_ONEVAL are supported
     *             FLOAT_ARG and STRING_ARG as accepted for backward compatibility.
     *             FLOAT_ARG is treated like FLOAT_ARRAY, and STRING_ARG like STRING_ONEVAL.
     * Note: the distiction between array values and single values is only important for
     *       pyferret: how to get the (string) values from ferret.  Ferret only expects to
     *       find FLOAT_ARG or STRING_ARG.
     */
    valobj = PyDict_GetItemString(initdict, "argtypes"); /* borrowed reference */
    if ( valobj != NULL ) {
        seqobj = PySequence_Fast(valobj, "argtypes value");
        if ( seqobj == NULL ) {
            PyErr_Clear();
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"argtypes\" value (not a tuple or list)");
            return;
        }
        seqlen = (int) PySequence_Fast_GET_SIZE(seqobj);
        if ( seqlen > num_args ) {
            Py_DECREF(seqobj);
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"argtypes\" value (tuple or list with too many items)");
            return;
        }
    }
    else {
        seqobj = NULL;
        seqlen = -1;
    }
    for (j = 0; j < num_args; j++) {
        /* FLOAT_ARG is the default */
        argtype = FLOAT_ARG;
        if ( j < seqlen ) {
            /* Get the type from the tuple */
            itemobj = PySequence_Fast_GET_ITEM(seqobj, (Py_ssize_t) j); /* borrowed reference */
#if PY_MAJOR_VERSION > 2
            val = (int) PyLong_AsLong(itemobj);
#else
            val = (int) PyInt_AsLong(itemobj);
#endif
            switch( val ) {
                case FLOAT_ONEVAL:
                case FLOAT_ARRAY:
                case FLOAT_ARG:
                    argtype = FLOAT_ARG;
                    break;
                case STRING_ONEVAL:
                case STRING_ARRAY:
                case STRING_ARG:
                    argtype = STRING_ARG;
                    break;
                default:
                    Py_DECREF(seqobj);
                    Py_DECREF(initdict);
                    sprintf(errmsg, "Unknown \"argtypes\" value for ARG%d", j+1);
                    return;
            }
        }
        q = j+1;
        FORTRAN(ef_set_arg_type)(&id, &q, &argtype);
    }
    Py_XDECREF(seqobj);

    /*
     * "influences": N-tuple of 6-tuples of booleans indicating whether the corresponding input
     *               argument's (X,Y,Z,T,E,F) axis influences the result grid's (X,Y,Z,T,E,F) axis.
     *               [optional, None for a 6-tuple and default: True for each value]
     */
    valobj = PyDict_GetItemString(initdict, "influences"); /* borrowed reference */
    if ( valobj != NULL ) {
        seqobj = PySequence_Fast(valobj, "influences value");
        if ( seqobj == NULL ) {
            PyErr_Clear();
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"influences\" value (not a tuple or list)");
            return;
        }
        seqlen = (int) PySequence_Fast_GET_SIZE(seqobj);
        if ( seqlen > num_args ) {
            Py_DECREF(seqobj);
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"influences\" value (tuple or list with too many items)");
            return;
        }
    }
    else {
        seqobj = NULL;
        seqlen = -1;
    }
    for (j = 0; j < num_args; j++) {
        for (k = 0; k < MAX_FERRET_NDIM; k++)
            axisvals[k] = YES;
        if ( j < seqlen ) {
            valobj = PySequence_Fast_GET_ITEM(seqobj, (Py_ssize_t) j); /* borrowed reference */
            /* None is acceptable here: treat as (True, True, True, True) */
            if ( valobj != Py_None ) {
                subseqobj = PySequence_Fast(valobj, "influences item");
                if ( subseqobj == NULL ) {
                    PyErr_Clear();
                    Py_DECREF(seqobj);
                    Py_DECREF(initdict);
                    sprintf(errmsg, "Invalid \"influences\" value (not None, a tuple, or a list) "
                                    "for ARG%d", j+1);
                    return;
                }
                subseqlen = (int) PySequence_Fast_GET_SIZE(subseqobj);
                if ( subseqlen > MAX_FERRET_NDIM ) {
                    Py_DECREF(subseqobj);
                    Py_DECREF(seqobj);
                    Py_DECREF(initdict);
                    sprintf(errmsg, "Invalid \"influences\" value (tuple or list with more than %d items) "
                                    "for ARG%d", MAX_FERRET_NDIM, j+1);
                    return;
                }
                for (k = 0; k < subseqlen; k++) {
                    itemobj = PySequence_Fast_GET_ITEM(subseqobj, (Py_ssize_t) k); /* borrowed reference */
                    /* Must be one of the singleton objects Py_True or Py_False to be accepted */
                    if ( itemobj == Py_False ) {
                        axisvals[k] = NO;
                    }
                    else if ( itemobj != Py_True ) {
                        PyErr_Clear();
                        Py_DECREF(subseqobj);
                        Py_DECREF(seqobj);
                        Py_DECREF(initdict);
                        sprintf(errmsg, "Invalid \"influences\" value (not True or False) for the %s axis "
                                        "of ARG%d", AXIS_NAMES[k], j+1);
                        return;
                    }
                }
                Py_DECREF(subseqobj);
            }
        }
        for (q = 0; q < MAX_FERRET_NDIM; q++) {
            if ( (axisvals[q] == YES) && (rsltaxes[q] != IMPLIED_BY_ARGS) && (rsltaxes[q] != REDUCED) ) {
                Py_DECREF(seqobj);
                Py_DECREF(initdict);
                sprintf(errmsg, "Invalid YES \"influences\" value (result axis not IMPLIED_BY_ARGS or REDUCED) "
                                "for the %s axis of ARG%d", AXIS_NAMES[k], j+1);
                return;
            }
        }
        q = j+1;
        FORTRAN(ef_set_axis_influence_6d)(&id, &q, &(axisvals[0]), &(axisvals[1]), &(axisvals[2]), 
                                           &(axisvals[3]), &(axisvals[4]), &(axisvals[5]));
    }
    Py_XDECREF(seqobj);

    /*
     * "extends": N-tuple of 6-tuples of pairs of integers.  The n-th tuple, if not None,
     *            gives the (X,Y,Z,T,E,F) extension pairs (which may be None) for the n-th
     *            argument.  [optional, default: no extensions assigned]
     *
     * Note: if not given, not assigned (instead of defaults assigned as above).
     */
    valobj = PyDict_GetItemString(initdict, "extends"); /* borrowed reference */
    if ( valobj != NULL ) {
        seqobj = PySequence_Fast(valobj, "extends value");
        if ( seqobj == NULL ) {
            PyErr_Clear();
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"extends\" value (not a tuple or list)");
            return;
        }
        seqlen = (int) PySequence_Fast_GET_SIZE(seqobj);
        if ( seqlen > num_args ) {
            Py_DECREF(seqobj);
            Py_DECREF(initdict);
            strcpy(errmsg, "Invalid \"extends\" value (tuple or list with too many items)");
            return;
        }
        for (j = 0; j < seqlen; j++) {
            valobj = PySequence_Fast_GET_ITEM(seqobj, (Py_ssize_t) j); /* borrowed reference */
	    /* None is acceptable here */
            if ( valobj != Py_None ) {
                subseqobj = PySequence_Fast(valobj, "extends item");
                if ( subseqobj == NULL ) {
                    PyErr_Clear();
                    Py_DECREF(seqobj);
                    Py_DECREF(initdict);
                    sprintf(errmsg, "Invalid \"extends\" value (not None, a tuple, or a list) "
                                    "for ARG%d", j+1);
                    return;
                }
                subseqlen = (int) PySequence_Fast_GET_SIZE(subseqobj);
                if ( subseqlen > MAX_FERRET_NDIM ) {
                    Py_DECREF(subseqobj);
                    Py_DECREF(seqobj);
                    Py_DECREF(initdict);
                    sprintf(errmsg, "Invalid \"extends\" value (tuple or list with more than %d items) "
                                    "for ARG%d", MAX_FERRET_NDIM, j+1);
                    return;
                }
                for (k = 0; k < subseqlen; k++) {
                    valobj = PySequence_Fast_GET_ITEM(subseqobj, (Py_ssize_t) k); /* borrowed reference */
	            /* None is acceptable here */
                    if ( valobj != Py_None ) {
                        subsubseqobj = PySequence_Fast(valobj, "extends item's item");
                        if ( subsubseqobj == NULL ) {
                            PyErr_Clear();
                            Py_DECREF(subseqobj);
                            Py_DECREF(seqobj);
                            Py_DECREF(initdict);
                            sprintf(errmsg, "Invalid \"extends\" value (not None, a tuple, or a list) "
                                            "for the %s axis of ARG%d", AXIS_NAMES[k], j+1);
                            return;
                        }
                        subsubseqlen = PySequence_Fast_GET_SIZE(subsubseqobj);
                        if ( subsubseqlen > 2 ) {
                            Py_DECREF(subsubseqobj);
                            Py_DECREF(subseqobj);
                            Py_DECREF(seqobj);
                            Py_DECREF(initdict);
                            sprintf(errmsg, "Invalid \"extends\" value (tuple or list with more that two items) "
                                            "for the %s axis of ARG%d", AXIS_NAMES[k], j+1);
                        }
                        deltas[0] = 0; deltas[1] = 0;
                        for (q = 0; q < subsubseqlen; q++) {
                            itemobj = PySequence_Fast_GET_ITEM(subsubseqobj, (Py_ssize_t) q); /* borrowed reference */
#if PY_MAJOR_VERSION > 2
                            val = PyLong_AsLong(itemobj);
#else
                            val = PyInt_AsLong(itemobj);
#endif
                            if ( PyErr_Occurred() ) {
                                PyErr_Clear();
                                Py_DECREF(subsubseqobj);
                                Py_DECREF(subseqobj);
                                Py_DECREF(seqobj);
                                Py_DECREF(initdict);
                                if ( q == 0 )
                                    sprintf(errmsg, "Invalid first \"extends\" value (not an int) "
                                                    "for the %s axis of ARG%d", AXIS_NAMES[k], j+1);
                                else
                                    sprintf(errmsg, "Invalid second \"extends\" value (not an int) "
                                                    "for the %s axis of ARG%d", AXIS_NAMES[k], j+1);
                                return;
                            }
                            deltas[q] = val;
                        }
                        if ( rsltaxes[k] != IMPLIED_BY_ARGS ) {
                            Py_DECREF(subsubseqobj);
                            Py_DECREF(subseqobj);
                            Py_DECREF(seqobj);
                            Py_DECREF(initdict);
                            sprintf(errmsg, "Invalid \"extends\" value (result axis not IMPLIED_BY_ARGS) "
                                            "for the %s axis of ARG%d", AXIS_NAMES[k], j+1);
                            return;
                        }
                        q = j+1;
                        val = k+1;
                        FORTRAN(ef_set_axis_extend)(&id, &q, &val, &(deltas[0]), &(deltas[1]));
                        Py_DECREF(subsubseqobj);
                    }
                }
                Py_DECREF(subseqobj);
            }
        }
        Py_DECREF(seqobj);
    }

    /* Iterate over all the keys to check for unknown keys (eg, typos) */
    keysobj = PyDict_Keys(initdict);
    seqobj = PySequence_Fast(keysobj, "dictionary keys");
    seqlen = (int) PySequence_Fast_GET_SIZE(seqobj);
    for (k = 0; k < seqlen; k++) {
        itemobj = PySequence_Fast_GET_ITEM(seqobj, (Py_ssize_t) k); /* borrowed reference */
#if PY_MAJOR_VERSION > 2
        strptr = PyUnicode_AsUTF8(itemobj);
#else
        strptr = PyString_AsString(itemobj);
#endif
        if ( strptr == NULL ) {
            Py_DECREF(seqobj);
            Py_DECREF(keysobj);
            Py_DECREF(initdict);
            sprintf(errmsg, "Invalid key (not a string) in the dictionary returned from %s in %s",
                            INIT_METHOD_NAME, modname);
            return;
        }
        if ( (strcmp(strptr, "numargs") != 0) && (strcmp(strptr, "descript") != 0) &&
             (strcmp(strptr, "restype") != 0) && (strcmp(strptr, "resstrlen") != 0) &&
             (strcmp(strptr, "axes") != 0) && (strcmp(strptr, "argnames") != 0) &&
             (strcmp(strptr, "argdescripts") != 0) && (strcmp(strptr, "argtypes") != 0) &&
             (strcmp(strptr, "influences") != 0) && (strcmp(strptr, "extends") != 0) &&
             (strcmp(strptr, "piecemeal") != 0) ) {
            sprintf(errmsg, "Invalid key \"%s\" in the dictionary returned from %s in %s",
                            strptr, INIT_METHOD_NAME, modname);
            Py_DECREF(seqobj);
            Py_DECREF(keysobj);
            Py_DECREF(initdict);
            return;
        }
    }
    Py_DECREF(seqobj);
    Py_DECREF(keysobj);

    /* initdict no longer needed */
    Py_DECREF(initdict);

    /* Success */
    errmsg[0] = '\0';
    return;
}

