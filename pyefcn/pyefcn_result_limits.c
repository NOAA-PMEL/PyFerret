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

#include "Python.h"
#include "numpy/arrayobject.h"
#include "pyefcn.h"

static const char *AXIS_NAMES[MAX_FERRET_NDIM] = { "X", "Y", "Z", "T" };

/*
 * See pyefcn.h for information on this function
 */
void pyefcn_result_limits(int id, char modname[], char errmsg[])
{
    PyObject  *valobj;
    PyObject  *usermod;
    PyObject  *seqobj;
    int        seqlen;
    int        k, q;
    int        call_made;
    PyObject  *subseqobj;
    int        subseqlen;
    PyObject  *itemobj;
    int        limits[2];

    /* Make sure Python is loaded in memory */
    Py_Initialize();

    /*
     * Import the user's Python module
     */
    valobj = PyString_FromString(modname);
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
     * Call the ferret_result_limits method in the user's python module with the ferret function ID as the sole argument
     */
    valobj = PyObject_CallMethod(usermod, RESULT_LIMITS_METHOD_NAME, "i", id);
    /* usermod no longer needed */
    Py_DECREF(usermod);
    /* check for errors */
    if ( valobj == NULL ) {
        sprintf(errmsg, "Error when calling %s in %s: %s", RESULT_LIMITS_METHOD_NAME, modname, pyefcn_get_error());
        return;
    }

    /*
     * Process the contents of the tuple returned, which cannot be None, since one of the axes needs to be assigned
     */
    seqobj = PySequence_Fast(valobj, "limits tuple");
    /* valobj no longer needed - PySequence_Fast has either incremented the reference count or made a copy as a tuple */
    Py_DECREF(valobj);
    if ( seqobj == NULL ) {
        PyErr_Clear();
        sprintf(errmsg, "Invalid return value (not a tuple or list) from %s in %s", RESULT_LIMITS_METHOD_NAME, modname);
        return;
    }
    seqlen = (int) PySequence_Fast_GET_SIZE(seqobj);
    if ( seqlen > MAX_FERRET_NDIM ) {
        Py_DECREF(seqobj);
        sprintf(errmsg, "Invalid return value (tuple or list with more than %d items) from %s in %s", 
                        MAX_FERRET_NDIM, RESULT_LIMITS_METHOD_NAME, modname);
        return;
    }

    /* Process each item in the tuple returned */
    call_made = 0;
    for (k = 0; k < seqlen; k++) {
        valobj = PySequence_Fast_GET_ITEM(seqobj, (Py_ssize_t) k); /* borrowed reference */
        /* None is acceptable here */
        if ( valobj != Py_None ) {
            subseqobj = PySequence_Fast(valobj, "limits item");
            if ( subseqobj == NULL ) {
                PyErr_Clear();
                Py_DECREF(seqobj);
                sprintf(errmsg, "Invalid result limits value (not None, a tuple, or a list) for the %s axis", AXIS_NAMES[k]);
                return;
            }
            subseqlen = (int) PySequence_Fast_GET_SIZE(subseqobj);
            /* If given, it must be a pair */
            if ( subseqlen != 2 ) {
                Py_DECREF(subseqobj);
                Py_DECREF(seqobj);
                sprintf(errmsg, "Invalid result limits value (not a pair of values) for the %s axis", AXIS_NAMES[k]);
                return;
            }
            for (q = 0; q < 2; q++) {
                itemobj = PySequence_Fast_GET_ITEM(subseqobj, (Py_ssize_t) q); /* borrowed reference */
                limits[q] = (int) PyInt_AsLong(itemobj);
                if ( PyErr_Occurred() ) {
                    PyErr_Clear();
                    Py_DECREF(subseqobj);
                    Py_DECREF(seqobj);
                    if ( q == 0 )
                        sprintf(errmsg, "Invalid result limits low value (not an int) for the %s axis", AXIS_NAMES[k]);
                    else
                        sprintf(errmsg, "Invalid result limits high value (not an int) for the %s axis", AXIS_NAMES[k]);
                    return;
                }
            }
            Py_DECREF(subseqobj);
            q = k+1;
            ef_set_axis_limits_(&id, &q, &(limits[0]), &(limits[1]));
            call_made = 1;
        }
    }
    Py_DECREF(seqobj);

    /* Make sure ef_set_axis_limits_ was called at least once */
    if ( ! call_made )
        sprintf(errmsg, "No result limits were given in the tuple returned from %s in %s", RESULT_LIMITS_METHOD_NAME, modname);
    else
        errmsg[0] = '\0';
    return;
}

