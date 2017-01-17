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
#include "pyferret.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* For older versions of NumPy (v1.4 with RHEL6), define this flag as the deprecated flag */
#ifndef NPY_ARRAY_OWNDATA
#define NPY_ARRAY_OWNDATA NPY_OWNDATA
#endif

/*
 * This function copies the data from the ndarray given by data_ndarray_ptr 
 * to the array of doubles given by dest.  The argument data_ndarray_ptr 
 * is a pointer to a PyObject pointer that is a float64 ndarray containing
 * the array of data for this static variable.
 */
void copy_pystat_data_(double dest[], void *data_ndarray_ptr)
{
    PyArrayObject *data_ndarray;
    double        *data;
    npy_intp       num_items;

    data_ndarray = *( (PyArrayObject **) data_ndarray_ptr);

    /* Sanity check:
     *    PyArray_Size returns 0 if the object is not an appropriate type
     *    ISFARRAY_RO checks if it is F-contiguous, aligned, and in machine byte-order 
     */
    num_items = PyArray_Size((PyObject *) data_ndarray);
    if ( (num_items < 1) || (PyArray_TYPE(data_ndarray) != NPY_DOUBLE) ||
         (! PyArray_ISFARRAY_RO(data_ndarray)) || (! PyArray_CHKFLAGS(data_ndarray, NPY_ARRAY_OWNDATA)) ) {
        fflush(stdout);
        fputs("Unexpected data_ndarray pointer passed to copy_pystat_data_\n", stderr);
        fflush(stderr);
        abort();
    }
    data = (double *) PyArray_DATA(data_ndarray);
    memcpy(dest, data, (size_t)num_items * sizeof(double));
}

