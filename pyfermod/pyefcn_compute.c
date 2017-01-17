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
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "pyferret.h"
#include "EF_Util.h"

/*
 * See pyferret.h for information on this function
 */
void pyefcn_compute(int id, char modname[], double *data[], int numarrays,
                    int memlo[][MAX_FERRET_NDIM], int memhi[][MAX_FERRET_NDIM],
                    int steplo[][MAX_FERRET_NDIM], int stephi[][MAX_FERRET_NDIM],
                    int incr[][MAX_FERRET_NDIM], double badvals[], char errmsg[])
{
    PyObject      *nameobj;
    PyObject      *usermod;
    PyObject      *initdict;
    PyObject      *typetuple;
    PyObject      *typeobj;
    int            j, k;
    int            datatypes[EF_MAX_COMPUTE_ARGS+1];
    int            resstrlen;
    npy_intp       shape[MAX_FERRET_NDIM];
    npy_intp       strides[MAX_FERRET_NDIM];
    int            itemsize;
    int            flags;
    double        *dataptr;
    int            maxlength;
    int            length;
    double        *dptr;
    npy_intp       d0, d1, d2, d3, d4, d5;
    npy_intp       indices[MAX_FERRET_NDIM];
    PyArrayObject *ndarrays[EF_MAX_COMPUTE_ARGS];
    PyArrayObject *inpbadvals_ndarray;
    PyArrayObject *resbadval_ndarray;
    PyObject      *idobj;
    PyObject      *inpobj;
    PyObject      *result;
    char          *strptr;

    /* Sanity check */
    if ( (numarrays < 2) || (numarrays > EF_MAX_COMPUTE_ARGS) ) {
        sprintf(errmsg, "Unexpected number of arrays (%d) passed to pyefcn_compute", numarrays);
        return;
    }

    /* Import the user's Python module */
#if PY_MAJOR_VERSION > 2
    nameobj = PyUnicode_FromString(modname);
#else
    nameobj = PyString_FromString(modname);
#endif
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
    /* Get the result type - default FLOAT_ARRAY */
    typeobj = PyDict_GetItemString(initdict, "restype"); /* borrowed reference */
    if ( typeobj != NULL ) {
#if PY_MAJOR_VERSION > 2
        datatypes[0] = (int) PyLong_AsLong(typeobj);
#else
        datatypes[0] = (int) PyInt_AsLong(typeobj);
#endif
    }
    else
        datatypes[0] = FLOAT_ARRAY;
    /* Get the (maximum) length of strings in a string result array - default 128 */
    typeobj = PyDict_GetItemString(initdict, "resstrlen"); /* borrowed reference */
    if ( typeobj != NULL ) {
#if PY_MAJOR_VERSION > 2
        resstrlen = (int) PyLong_AsLong(typeobj);
#else
        resstrlen = (int) PyInt_AsLong(typeobj);
#endif
    }
    else
        resstrlen = 128;
    /* Find out the argument types */
    typetuple = PyDict_GetItemString(initdict, "argtypes"); /* borrowed reference */
    /* If typetuple is NULL, the key is not present but no error was raised */
    j = 1;
    if ( typetuple != NULL ) {
        for ( ; j < numarrays; j++) {
            /* Get the type of this argument */
            typeobj = PySequence_GetItem(typetuple, (Py_ssize_t) (j-1));
            if ( typeobj == NULL ) {
                PyErr_Clear();
                break;
            }
#if PY_MAJOR_VERSION > 2
            datatypes[j] = (int) PyLong_AsLong(typeobj);
#else
            datatypes[j] = (int) PyInt_AsLong(typeobj);
#endif
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
                /* Get the strides through the passed memory as a (double *) */
                strides[0] = 1;
                for (k = 0; k < MAX_FERRET_NDIM - 1; k++)
                    strides[k+1] = strides[k] * (npy_intp) (memhi[j][k] - memlo[j][k] + 1);
                /* Get the actual starting point in the array */
                dataptr = data[j];
                for (k = 0; k < MAX_FERRET_NDIM; k++)
                    dataptr += strides[k] * (npy_intp) (steplo[j][k] - memlo[j][k]);
                /* Convert to strides through places in memory to be assigned, and as a (byte *) */
                itemsize = sizeof(double);
                for (k = 0; k < MAX_FERRET_NDIM; k++)
                    strides[k] *= (npy_intp) (incr[j][k] * itemsize);
                /* Get the flags for the array - only results can be written to; others are read-only */
                for (k = 0; k < MAX_FERRET_NDIM; k++)
                    if ( (incr[j][k] != 1) || (steplo[j][k] != memlo[j][k]) )
                        break;
                if ( k < MAX_FERRET_NDIM )
                    flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
                else
                    flags = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
                if ( j == 0 )
                    flags = flags | NPY_ARRAY_WRITEABLE;
                /* Create a PyArray object around the array */
                ndarrays[j] = (PyArrayObject *) PyArray_New(&PyArray_Type, MAX_FERRET_NDIM, shape, NPY_DOUBLE,
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
                /* Simple double argument; just create a PyFloat for this argument */
                ndarrays[j] = (PyArrayObject *) PyFloat_FromDouble(data[j][0]);
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
                /* Get the dimensions of the array */
                for (k = 0; k < MAX_FERRET_NDIM; k++)
                    shape[k] = (npy_intp) ((stephi[j][k] - steplo[j][k] + incr[j][k]) / (incr[j][k]));
                /* Get the strides through the passed memory as a (double *) */
                strides[0] = 1;
                for (k = 0; k < MAX_FERRET_NDIM - 1; k++)
                    strides[k+1] = strides[k] * (npy_intp) (memhi[j][k] - memlo[j][k] + 1);
                if ( j == 0 ) {
                    /* result argument - create PyArray of string to hold results to be assigned */
                    itemsize = resstrlen * sizeof(char);
                    ndarrays[j] = (PyArrayObject *) PyArray_New(&PyArray_Type, MAX_FERRET_NDIM, shape, NPY_STRING,
                                              NULL, NULL, itemsize, NPY_ARRAY_FARRAY, NULL);
                    if ( ndarrays[j] == NULL ) {
                        /* Problem - release references to the previous PyArray objects, assign errmsg, and return */
                        PyErr_Clear();
                        sprintf(errmsg, "Unable to create ndarray[%d]", j);
                        /* First array creation attempt  - no other ndarray element */
                        Py_DECREF(usermod);
                        return;
                    }
                }
                else {
                    /* Get the actual starting point in the array */
                    dataptr = data[j];
                    for (k = 0; k < MAX_FERRET_NDIM; k++)
                        dataptr += strides[k] * (npy_intp) (steplo[j][k] - memlo[j][k]);
                    /* Input argument - get the length of the longest string */
                    /* This needs to be modified if MAX_FERRET_NDIM changes */
                    maxlength = 0;
                    for (d5 = 0; d5 < shape[5] * strides[5]; d5 += strides[5]) {
                      for (d4 = 0; d4 < shape[4] * strides[4]; d4 += strides[4]) {
                        for (d3 = 0; d3 < shape[3] * strides[3]; d3 += strides[3]) {
                          for (d2 = 0; d2 < shape[2] * strides[2]; d2 += strides[2]) {
                            for (d1 = 0; d1 < shape[1] * strides[1]; d1 += strides[1]) {
                              for (d0 = 0; d0 < shape[0] * strides[0]; d0 += strides[0]) {
                                /*
                                 * The data array values are pointers to strings,
                                 * but is cast as an array of doubles
                                 */
                                dptr = dataptr + d0 + d1 + d2 + d3 + d4 + d5;
                                length = strlen(*((char **) dptr));
                                if ( maxlength < length )
                                    maxlength = length;
                              }
                            }
                          }
                        }
                      }
                    }
                    /* Convert to the next larger multiple of 8 */
                    maxlength  = (maxlength + 8) / 8;
                    maxlength *= 8;
                    /* Create a PyArray object of strings to hold a copy of the data */
                    itemsize = maxlength * sizeof(char);
                    ndarrays[j] = (PyArrayObject *) PyArray_New(&PyArray_Type, MAX_FERRET_NDIM, shape, NPY_STRING,
                                              NULL, NULL, itemsize, NPY_ARRAY_FARRAY_RO, NULL);
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
                    /* This needs to be modified if MAX_FERRET_NDIM changes */
                    indices[5] = 0;
                    for (d5 = 0; d5 < shape[5] * strides[5]; d5 += strides[5]) {
                      indices[4] = 0;
                      for (d4 = 0; d4 < shape[4] * strides[4]; d4 += strides[4]) {
                        indices[3] = 0;
                        for (d3 = 0; d3 < shape[3] * strides[3]; d3 += strides[3]) {
                          indices[2] = 0;
                          for (d2 = 0; d2 < shape[2] * strides[2]; d2 += strides[2]) {
                            indices[1] = 0;
                            for (d1 = 0; d1 < shape[1] * strides[1]; d1 += strides[1]) {
                              indices[0] = 0;
                              for (d0 = 0; d0 < shape[0] * strides[0]; d0 += strides[0]) {
                                dptr = dataptr + d0 + d1 + d2 + d3 + d4 + d5;
                                strcpy((char *) PyArray_GetPtr(ndarrays[j], indices), *((char **) dptr));
                                (indices[0])++;
                              }
                              (indices[1])++;
                            }
                            (indices[2])++;
                          }
                          (indices[3])++;
                        }
                        (indices[4])++;
                      }
                      (indices[5])++;
                    }

                }
                break;
            case STRING_ONEVAL:
            case STRING_ARG:
                assert( j > 0 );
                /* String argument; just create a PyUnicode/PyString for this argument */
#if PY_MAJOR_VERSION > 2
                ndarrays[j] = (PyArrayObject *) PyUnicode_FromString(*((char **) (data[j])));
#else
                ndarrays[j] = (PyArrayObject *) PyString_FromString(*((char **) (data[j])));
#endif
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
        PyTuple_SET_ITEM(inpobj, (Py_ssize_t)(j-1), (PyObject *) ndarrays[j]); /* Steals a reference to ndarrays[j] */
    }

    /* Create PyArray objects around the input bad values array and the result bad value */
    shape[0] = numarrays - 1;
    strides[0] = sizeof(double);
    itemsize = sizeof(double);
    flags = NPY_ARRAY_FARRAY_RO;
    inpbadvals_ndarray = (PyArrayObject *) PyArray_New(&PyArray_Type, 1, shape, NPY_DOUBLE,
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
    resbadval_ndarray = (PyArrayObject *) PyArray_New(&PyArray_Type, 1, shape, NPY_DOUBLE,
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
#if PY_MAJOR_VERSION > 2
    idobj = PyLong_FromLong((long)id);
#else
    idobj = PyInt_FromLong((long)id);
#endif

    /* Call the ferret_compute function in the module */
#if PY_MAJOR_VERSION > 2
    nameobj = PyUnicode_FromString(COMPUTE_METHOD_NAME);
#else
    nameobj = PyString_FromString(COMPUTE_METHOD_NAME);
#endif
    result = PyObject_CallMethodObjArgs(usermod, nameobj, idobj, ndarrays[0], resbadval_ndarray,
                                                                 inpobj, inpbadvals_ndarray, NULL);

    /* Release all the PyObjects no longer needed */
    Py_XDECREF(result);
    Py_DECREF(nameobj);
    Py_DECREF(idobj);
    Py_DECREF(resbadval_ndarray);
    Py_DECREF(inpbadvals_ndarray);
    Py_DECREF(inpobj);
    Py_DECREF(usermod);

    /* If the ferret_compute call was unsuccessful (raised an exception), assign errmsg from its message */
    if ( result == NULL ) {
        sprintf(errmsg, "Error when calling %s in %s: %s", COMPUTE_METHOD_NAME, modname, pyefcn_get_error());
        Py_DECREF(ndarrays[0]);
        return;
    }

    /*
     * If the return type is an array of strings, assign
     * the Ferret array with copies of the NumPy strings
     */
    if ( datatypes[0] == STRING_ARRAY ) {
        /* Get the dimensions of the array */
        for (k = 0; k < MAX_FERRET_NDIM; k++)
            shape[k] = (npy_intp) ((stephi[0][k] - steplo[0][k] + incr[0][k]) / (incr[0][k]));
        /* Get the strides through the passed memory as a (double *) */
        strides[0] = 1;
        for (k = 0; k <= MAX_FERRET_NDIM - 1; k++)
            strides[k+1] = strides[k] * (npy_intp) (memhi[0][k] - memlo[0][k] + 1);
        /* Get the actual starting point in the array */
        dataptr = data[0];
        for (k = 0; k < MAX_FERRET_NDIM; k++)
            dataptr += strides[k] * (npy_intp) (steplo[0][k] - memlo[0][k]);
        /* Assign all the strings in the array */
        /* This needs to be modified if MAX_FERRET_NDIM changes */
        indices[5] = 0;
        for (d5 = 0; d5 < shape[5] * strides[5]; d5 += strides[5]) {
          indices[4] = 0;
          for (d4 = 0; d4 < shape[4] * strides[4]; d4 += strides[4]) {
            indices[3] = 0;
            for (d3 = 0; d3 < shape[3] * strides[3]; d3 += strides[3]) {
              indices[2] = 0;
              for (d2 = 0; d2 < shape[2] * strides[2]; d2 += strides[2]) {
                indices[1] = 0;
                for (d1 = 0; d1 < shape[1] * strides[1]; d1 += strides[1]) {
                  indices[0] = 0;
                  for (d0 = 0; d0 < shape[0] * strides[0]; d0 += strides[0]) {
                    strptr = (char *) PyArray_GetPtr(ndarrays[0], indices);
                    for (j = 0; j < resstrlen; j++)
                        if ( strptr[j] == '\0' )
                            break;
                    dptr = dataptr + d0 + d1 + d2 + d3 + d4 + d5;
                    ef_put_string_(strptr, &j, (char **) dptr);
                    (indices[0])++;
                  }
                  (indices[1])++;
                }
                (indices[2])++;
              }
              (indices[3])++;
            }
            (indices[4])++;
          }
          (indices[5])++;
        }

    }
    Py_DECREF(ndarrays[0]);

    errmsg[0] = '\0';
    return;
}

