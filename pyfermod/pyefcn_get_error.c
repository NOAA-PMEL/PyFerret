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
#include "ferret.h"
#include "pyferret.h"

/*
 * See pyferret.h for information on this function
 */
char *pyefcn_get_error()
{
    /* returned error string thus must be static */
    static char errmsg[512];

    PyObject *exc_type;
    PyObject *exc_value;
    PyObject *exc_traceback;
    PyObject *exc_string;

    /* Initialize errmsg to no message */
    errmsg[0] = '\0';

    /* Check for an exception */
    PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
    if ( exc_type != NULL ) {
        /* Exception found and cleared - first normalize exc_value */
        PyErr_NormalizeException(&exc_type, &exc_value, &exc_traceback);
        /* Now exc_value is guaranteed to be an Exception object of class (or subclass of) exc_type */
        if ( exc_value != NULL ) {
            /* Get the string by calling Python str method with the exception */
            exc_string = PyObject_Str(exc_value);
            if ( exc_string != NULL ) {
#if PY_MAJOR_VERSION > 2
                strcpy(errmsg, PyUnicode_AsUTF8(exc_string));
#else
                strcpy(errmsg, PyString_AsString(exc_string));
#endif
                Py_DECREF(exc_string);
            }
        }
        /* Since there was an exception, make sure errmsg is not empty */
        if ( errmsg[0] == '\0' ) {
            strcpy(errmsg, "Exception raised with no message");
        }
    }

    /* Decrement references to non-NULL objects */
    Py_XDECREF(exc_type);
    Py_XDECREF(exc_value);
    Py_XDECREF(exc_traceback);

    return errmsg;
}

