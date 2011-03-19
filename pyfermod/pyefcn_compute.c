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
#include <numpy/arrayobject.h>
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
    int j, k;
    npy_intp shape[MAX_FERRET_NDIM];
    npy_intp strides[MAX_FERRET_NDIM];
    float *datptr;
    int itemsize;
    int datatype;
    int flags;
    PyObject *ndarrays[EF_MAX_COMPUTE_ARGS];
    PyObject *inpbadvals_ndarray;
    PyObject *resbadval_ndarray;
    PyObject *idobj;
    PyObject *inpobj;
    PyObject *result;

    /* Initialize Python and Numpy if needed */
    Py_Initialize();
    import_array();

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

    /* Create PyArray objects around the Fortran data arrays */
    for (j = 0; j < numarrays; j++) {
        /* Gets the dimensions of the array */
        for (k = 0; k < MAX_FERRET_NDIM; k++)
            shape[k] = (npy_intp) ((stephi[j][k] - steplo[j][k] + incr[j][k]) / (incr[j][k]));
        /* Get the strides through the passed memory as a (float *) */
        strides[0] = 1;
        for (k = 0; k < 3; k++)
            strides[k+1] = strides[k] * (npy_intp) (memhi[j][k] - memlo[j][k] + 1);
        /* Get the actual starting point in the array */
        for (k = 0; k < MAX_FERRET_NDIM; k++)
            data[j] += strides[k] * (npy_intp) (steplo[j][k] - memlo[j][k]);
        /* Convert to strides through places in memory to be assigned, and as a (byte *) */
        datatype = NPY_FLOAT;
        itemsize = sizeof(float);
        for (k = 0; k < MAX_FERRET_NDIM; k++)
            strides[k] *= (npy_intp) (incr[j][k] * itemsize);
        /* Get the flags for the array - only the results array can be written to; the others are read-only */
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
        ndarrays[j] = PyArray_New(&PyArray_Type, MAX_FERRET_NDIM, shape, datatype, strides, data[j], itemsize, flags, NULL);
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
    }

    /* Create a tuple with all the input arrays */
    inpobj = PyTuple_New((Py_ssize_t) (numarrays-1));
    for (j = 1; j < numarrays; j++) {
        PyTuple_SET_ITEM(inpobj, (Py_ssize_t)(j-1), ndarrays[j]); /* Steals a reference to ndarrays[j] */
    }

    /* Create PyArray objects around the input bad values array and the result bad value */
    shape[0] = numarrays - 1;
    datatype = NPY_FLOAT;
    strides[0] = sizeof(float);
    itemsize = sizeof(float);
    flags = NPY_FARRAY_RO;
    inpbadvals_ndarray = PyArray_New(&PyArray_Type, 1, shape, datatype, strides, &(badvals[1]), itemsize, flags, NULL);
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
    resbadval_ndarray = PyArray_New(&PyArray_Type, 1, shape, datatype, strides, badvals, itemsize, flags, NULL);
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
    result = PyObject_CallMethodObjArgs(usermod, nameobj, idobj, ndarrays[0], resbadval_ndarray, inpobj, inpbadvals_ndarray, NULL);

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
