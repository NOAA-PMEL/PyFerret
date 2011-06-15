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
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "pyferret.h"
#include "EF_Util.h"

/*
 * See pyferret.h for information on this function
 */
void pyefcn_compute(int id, char modname[], float *data[], int numarrays,
                    int memlo[][MAX_FERRET_NDIM], int memhi[][MAX_FERRET_NDIM],
                    int steplo[][MAX_FERRET_NDIM], int stephi[][MAX_FERRET_NDIM],
                    int incr[][MAX_FERRET_NDIM], float badvals[], char errmsg[])
{
    PyObject *nameobj;
    PyObject *usermod;
    PyObject *initdict;
    PyObject *typetuple;
    PyObject *typeobj;
    int       j, k;
    int       datatypes[EF_MAX_COMPUTE_ARGS+1];
    npy_intp  shape[MAX_FERRET_NDIM];
    npy_intp  strides[MAX_FERRET_NDIM];
    int       itemsize;
    int       flags;
    float    *dataptr;
    int       maxlength;
    int       length;
    double   *dptr;
    npy_intp  d0, d1, d2, d3;
    npy_intp  indices[4];
    PyObject *ndarrays[EF_MAX_COMPUTE_ARGS];
    PyObject *inpbadvals_ndarray;
    PyObject *resbadval_ndarray;
    PyObject *idobj;
    PyObject *inpobj;
    PyObject *result;

    /* Sanity check */
    if ( (numarrays < 2) || (numarrays > EF_MAX_COMPUTE_ARGS) ) {
        sprintf(errmsg, "Unexpected number of arrays (%d) passed to pyefcn_compute", numarrays);
        return;
    }

    /* Import the user's Python module */
    nameobj = PyString_FromString(modname);
    if ( nameobj == NULL ) {
        PyErr_Clear();
        sprintf(errmsg, "Problems creating a Python string from the module name: %s", modname);
        return;
    }
    usermod = PyImport_Import(nameobj);
    Py_DECREF(nameobj);
    if ( usermod == NULL ) {
        PyErr_Clear();
        sprintf(errmsg, "Unable to import module: %s", modname);
        return;
    }

    /* Call the initialization method to find out the data types */
    initdict = PyObject_CallMethod(usermod, INIT_METHOD_NAME, "i", id);
    if ( initdict == NULL ) {
        Py_DECREF(usermod);
        sprintf(errmsg, "Error when calling %s in %s: %s", INIT_METHOD_NAME, modname, pyefcn_get_error());
        return;
    }
    /* Currently the result has to be FLOAT_ARRAY */
    datatypes[0] = FLOAT_ARRAY;
    /* Find out the argument types */
    typetuple = PyDict_GetItemString(initdict, "argtypes"); /* borrowed reference */
    /* If typetuple is NULL, the key is not present but no exception was raised */
    j = 1;
    if ( typetuple != NULL ) {
        for ( ; j < numarrays; j++) {
            /* Get the type of this argument */
            typeobj = PySequence_GetItem(typetuple, (Py_ssize_t) (j-1));
            if ( typeobj == NULL ) {
                PyErr_Clear();
                break;
            }
            datatypes[j] = (int) PyInt_AsLong(typeobj);
            Py_DECREF(typeobj);
        }
    }
    /* Assign the default FLOAT_ARRAY for any unspecified types */
    for ( ; j < numarrays; j++)
        datatypes[j] = FLOAT_ARRAY;
    Py_DECREF(initdict);

    /* Create the Python objects for the inputs and result. */
    for (j = 0; j < numarrays; j++) {
        switch( datatypes[j] ) {
            case FLOAT_ARRAY:
            case FLOAT_ARG:
                /* Get the dimensions of the array */
                for (k = 0; k < MAX_FERRET_NDIM; k++)
                    shape[k] = (npy_intp) ((stephi[j][k] - steplo[j][k] + incr[j][k]) / (incr[j][k]));
                /* Get the strides through the passed memory as a (float *) */
                strides[0] = 1;
                for (k = 0; k < 3; k++)
                    strides[k+1] = strides[k] * (npy_intp) (memhi[j][k] - memlo[j][k] + 1);
                /* Get the actual starting point in the array */
                dataptr = data[j];
                for (k = 0; k < MAX_FERRET_NDIM; k++)
                    dataptr += strides[k] * (npy_intp) (steplo[j][k] - memlo[j][k]);
                /* Convert to strides through places in memory to be assigned, and as a (byte *) */
                itemsize = sizeof(float);
                for (k = 0; k < MAX_FERRET_NDIM; k++)
                    strides[k] *= (npy_intp) (incr[j][k] * itemsize);
                /* Get the flags for the array - only results can be written to; others are read-only */
                for (k = 0; k < MAX_FERRET_NDIM; k++)
                    if ( (incr[j][k] != 1) || (steplo[j][k] != memlo[j][k]) )
                        break;
                if ( k < MAX_FERRET_NDIM )
                    flags = NPY_ALIGNED | NPY_NOTSWAPPED;
                else
                    flags = NPY_F_CONTIGUOUS | NPY_ALIGNED | NPY_NOTSWAPPED;
                if ( j == 0 )
                    flags = flags | NPY_WRITEABLE;
                /* Create a PyArray object around the array */
                ndarrays[j] = PyArray_New(&PyArray_Type, MAX_FERRET_NDIM, shape, NPY_FLOAT,
                                          strides, dataptr, itemsize, flags, NULL);
                if ( ndarrays[j] == NULL ) {
                    /* Problem - release references to the previous PyArray objects, assign errmsg, and return */
                    PyErr_Clear();
                    sprintf(errmsg, "Unable to create ndarray[%d]", j);
                    while ( j > 0 ) {
                        j--;
                        Py_DECREF(ndarrays[j]);
                    }
                    Py_DECREF(usermod);
                    return;
                }
                break;
            case FLOAT_ONEVAL:
                assert( j > 0 );
                /* Sinple float argument; just create a PyFloat for this argument */
                ndarrays[j] = PyFloat_FromDouble((double) (data[j][0]));
                if ( ndarrays[j] == NULL ) {
                    /* Problems - Release references to the previous PyArray objects, assign errmsg, and return. */
                    PyErr_Clear();
                    sprintf(errmsg, "Problems creating a Python float from input argument %d", j);
                    while ( j > 0 ) {
                        j--;
                        Py_DECREF(ndarrays[j]);
                    }
                    Py_DECREF(usermod);
                    return;
                }
                break;
            case STRING_ARRAY:
                assert( j > 0 );
                /* Get the dimensions of the array */
                for (k = 0; k < MAX_FERRET_NDIM; k++)
                    shape[k] = (npy_intp) ((stephi[j][k] - steplo[j][k] + incr[j][k]) / (incr[j][k]));
                /* Get the strides through the passed memory as a (double *) */
                strides[0] = 1;
                for (k = 0; k < 3; k++)
                    strides[k+1] = strides[k] * (npy_intp) (memhi[j][k] - memlo[j][k] + 1);
                /* Get the actual starting point in the array */
                dptr = (double *) (data[j]);
                for (k = 0; k < MAX_FERRET_NDIM; k++)
                    dptr += strides[k] * (npy_intp) (steplo[j][k] - memlo[j][k]);
                dataptr = (float *) dptr;
                /* Get the length of the longest string */
                maxlength = 0;
                for (d3 = 0; d3 < shape[3] * strides[3]; d3 += strides[3]) {
                    for (d2 = 0; d2 < shape[2] * strides[2]; d2 += strides[2]) {
                        for (d1 = 0; d1 < shape[1] * strides[1]; d1 += strides[1]) {
                            for (d0 = 0; d0 < shape[0] * strides[0]; d0 += strides[0]) {
                                /*
                                 * The data array values are pointers to strings,
                                 * but is cast as an array of doubles
                                 */
                                dptr = ((double *) dataptr) + d0 + d1 + d2 + d3;
                                length = strlen(*((char **) dptr));
                                if ( maxlength < length )
                                    maxlength = length;
                            }
                        }
                    }
                }
                /* Convert to the next larger multiple of 8 */
                maxlength  = (maxlength + 8) / 8;
                maxlength *= 8;
                /* Create a PyArray object of strings to hold a copy of the data */
                itemsize = maxlength * sizeof(char);
                ndarrays[j] = PyArray_New(&PyArray_Type, MAX_FERRET_NDIM, shape, NPY_STRING,
                                          NULL, NULL, itemsize, NPY_FARRAY_RO, NULL);
                if ( ndarrays[j] == NULL ) {
                    /* Problem - release references to the previous PyArray objects, assign errmsg, and return */
                    PyErr_Clear();
                    sprintf(errmsg, "Unable to create ndarray[%d]", j);
                    while ( j > 0 ) {
                        j--;
                        Py_DECREF(ndarrays[j]);
                    }
                    Py_DECREF(usermod);
                    return;
                }
                /* Assign all the strings in the array */
                indices[3] = 0;
                for (d3 = 0; d3 < shape[3] * strides[3]; d3 += strides[3]) {
                    indices[2] = 0;
                    for (d2 = 0; d2 < shape[2] * strides[2]; d2 += strides[2]) {
                        indices[1] = 0;
                        for (d1 = 0; d1 < shape[1] * strides[1]; d1 += strides[1]) {
                            indices[0] = 0;
                            for (d0 = 0; d0 < shape[0] * strides[0]; d0 += strides[0]) {
                                dptr = ((double *) dataptr) + d0 + d1 + d2 + d3;
                                strcpy((char *) PyArray_GetPtr(ndarrays[j], indices), *((char **) dptr));
                                (indices[0])++;
                            }
                            (indices[1])++;
                        }
                        (indices[2])++;
                    }
                    (indices[3])++;
                }
                break;
            case STRING_ONEVAL:
            case STRING_ARG:
                assert( j > 0 );
                /* String argument; just create a PyString for this argument */
                ndarrays[j] = PyString_FromString(*((char **) (data[j])));
                if ( ndarrays[j] == NULL ) {
                    /* Problems - Release references to the previous PyArray objects, assign errmsg, and return. */
                    PyErr_Clear();
                    sprintf(errmsg, "Problems creating a Python string from input argument %d", j);
                    while ( j > 0 ) {
                        j--;
                        Py_DECREF(ndarrays[j]);
                    }
                    Py_DECREF(usermod);
                    return;
                }
                break;
            default:
                /* Unknown type - Release references to the previous PyArray objects, assign errmsg, and return. */
                PyErr_Clear();
                sprintf(errmsg, "Unexpected error: unknown datatypes[%d] of %d", j, datatypes[j]);
                while ( j > 0 ) {
                    j--;
                    Py_DECREF(ndarrays[j]);
                }
                Py_DECREF(usermod);
                return;
        }
    }

    /* Create a tuple with all the input arrays */
    inpobj = PyTuple_New((Py_ssize_t) (numarrays-1));
    for (j = 1; j < numarrays; j++) {
        PyTuple_SET_ITEM(inpobj, (Py_ssize_t)(j-1), ndarrays[j]); /* Steals a reference to ndarrays[j] */
    }

    /* Create PyArray objects around the input bad values array and the result bad value */
    shape[0] = numarrays - 1;
    strides[0] = sizeof(float);
    itemsize = sizeof(float);
    flags = NPY_FARRAY_RO;
    inpbadvals_ndarray = PyArray_New(&PyArray_Type, 1, shape, NPY_FLOAT,
                                     strides, &(badvals[1]), itemsize, flags, NULL);
    if ( inpbadvals_ndarray == NULL ) {
        /* Problem - release references to the previous PyArray objects, assign errmsg, and return */
        PyErr_Clear();
        Py_DECREF(inpobj);
        Py_DECREF(ndarrays[0]);
        Py_DECREF(usermod);
        strcpy(errmsg, "Unable to create input badvals ndarray");
        return;
    }
    shape[0] = 1;
    resbadval_ndarray = PyArray_New(&PyArray_Type, 1, shape, NPY_FLOAT,
                                    strides, badvals, itemsize, flags, NULL);
    if ( resbadval_ndarray == NULL ) {
        /* Problem - release references to the previous PyArray objects, assign errmsg, and return */
        PyErr_Clear();
        Py_DECREF(inpbadvals_ndarray);
        Py_DECREF(inpobj);
        Py_DECREF(ndarrays[0]);
        Py_DECREF(usermod);
        strcpy(errmsg, "Unable to create result badvals ndarray");
        return;
    }

    /* ferret ID argument */
    idobj = PyInt_FromLong((long)id);

    /* Call the ferret_compute function in the module */
    nameobj = PyString_FromString(COMPUTE_METHOD_NAME);
    result = PyObject_CallMethodObjArgs(usermod, nameobj, idobj, ndarrays[0], resbadval_ndarray,
                                                                 inpobj, inpbadvals_ndarray, NULL);

    /* Release all the PyObjects no longer needed */
    Py_XDECREF(result);
    Py_DECREF(nameobj);
    Py_DECREF(idobj);
    Py_DECREF(resbadval_ndarray);
    Py_DECREF(inpbadvals_ndarray);
    Py_DECREF(inpobj);
    Py_DECREF(ndarrays[0]);
    Py_DECREF(usermod);

    /* If the ferret_compute call was unsuccessful (raised an exception), assign errmsg from its message */
    if ( result == NULL )
        sprintf(errmsg, "Error when calling %s in %s: %s", COMPUTE_METHOD_NAME, modname, pyefcn_get_error());
    else
        errmsg[0] = '\0';

    return;
}
