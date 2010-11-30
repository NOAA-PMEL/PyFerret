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
#include "EF_Util.h"

static const char *AXIS_NAMES[MAX_FERRET_NDIM] = { "X", "Y", "Z", "T" };

/*
 * See pyefcn.h for information on this function
 */
void pyefcn_custom_axes(int id, char modname[], char errmsg[])
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
    float      values[3];
    char      *strptr;
    char       unit_name[EF_MAX_NAME_LENGTH];
    int        is_modulo;

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
     * Call the ferret_custom_axes method in the user's python module with the ferret function ID as the sole argument
     */
    valobj = PyObject_CallMethod(usermod, CUSTOM_AXES_METHOD_NAME, "i", id);
    /* usermod no longer needed */
    Py_DECREF(usermod);
    /* check for errors */
    if ( valobj == NULL ) {
        sprintf(errmsg, "Error when calling %s in %s: %s", CUSTOM_AXES_METHOD_NAME, modname, pyefcn_get_error());
        return;
    }

    /*
     * Process the contents of the tuple returned, which cannot be None, since one of the axes needs to be assigned
     */
    seqobj = PySequence_Fast(valobj, "custom axes tuple");
    /* valobj no longer needed - PySequence_Fast has either incremented the reference count or made a copy as a tuple */
    Py_DECREF(valobj);
    if ( seqobj == NULL ) {
        PyErr_Clear();
        sprintf(errmsg, "Invalid return value (not a tuple or list) from %s in %s", CUSTOM_AXES_METHOD_NAME, modname);
        return;
    }
    seqlen = (int) PySequence_Fast_GET_SIZE(seqobj);
    if ( seqlen > MAX_FERRET_NDIM ) {
        Py_DECREF(seqobj);
        sprintf(errmsg, "Invalid return value (tuple or list with more than %d items) from %s in %s", 
                        MAX_FERRET_NDIM, CUSTOM_AXES_METHOD_NAME, modname);
        return;
    }

    /* Process each item in the tuple returned */
    call_made = 0;
    for (k = 0; k < seqlen; k++) {
        valobj = PySequence_Fast_GET_ITEM(seqobj, (Py_ssize_t) k); /* borrowed reference */
        /* None is acceptable here */
        if ( valobj != Py_None ) {
            subseqobj = PySequence_Fast(valobj, "custom axes item");
            if ( subseqobj == NULL ) {
                PyErr_Clear();
                Py_DECREF(seqobj);
                sprintf(errmsg, "Invalid custom axes value (not None, a tuple, or a list) for the %s axis", AXIS_NAMES[k]);
                return;
            }
            subseqlen = (int) PySequence_Fast_GET_SIZE(subseqobj);
            /* If given, it must have at least three items */
            if ( subseqlen < 3 ) {
                Py_DECREF(subseqobj);
                Py_DECREF(seqobj);
                sprintf(errmsg, "Invalid custom axes value (not a tuple of at least three values) for the %s axis", AXIS_NAMES[k]);
                return;
            }
            /* Get the low, high, delta floating point values */
            for (q = 0; q < 3; q++) {
                itemobj = PySequence_Fast_GET_ITEM(subseqobj, (Py_ssize_t) q); /* borrowed reference */
                values[q] = (float) PyFloat_AsDouble(itemobj);
                if ( PyErr_Occurred() ) {
                    PyErr_Clear();
                    Py_DECREF(subseqobj);
                    Py_DECREF(seqobj);
                    if ( q == 0 )
                        sprintf(errmsg, "Invalid custom axes low value (not an float) for the %s axis", AXIS_NAMES[k]);
                    else if ( q == 1 )
                        sprintf(errmsg, "Invalid custom axes high value (not an float) for the %s axis", AXIS_NAMES[k]);
                    else
                        sprintf(errmsg, "Invalid custom axes delta value (not an float) for the %s axis", AXIS_NAMES[k]);
                    return;
                }
            }
            /* Get the unit name, if given */
            strcpy(unit_name, " ");
            if ( subseqlen > 3 ) {
                itemobj = PySequence_Fast_GET_ITEM(subseqobj, (Py_ssize_t) 3); /* borrowed reference */
                strptr = PyString_AsString(itemobj);
                if ( strptr == NULL ) {
                    PyErr_Clear();
                    Py_DECREF(subseqobj);
                    Py_DECREF(seqobj);
                    sprintf(errmsg, "Invalid custom axes unit_name value (not a string) for the %s axis", AXIS_NAMES[k]);
                    return;
                }
                if ( strptr[0] != '\0' ) {
                    strncpy(unit_name, strptr, EF_MAX_NAME_LENGTH);
                    unit_name[EF_MAX_NAME_LENGTH-1] = '\0';
                }
            }
            /* get the is_modulo value, if given */
            is_modulo = 0;
            if ( subseqlen > 4 ) {
                itemobj = PySequence_Fast_GET_ITEM(subseqobj, (Py_ssize_t) 4); /* borrowed reference */
                /* must being either the Py_False singleton or the Py_True singleton */
                if ( itemobj == Py_True ) {
                    is_modulo = 1;
                }
                else if ( itemobj != Py_False ) {
                    PyErr_Clear();
                    Py_DECREF(subseqobj);
                    Py_DECREF(seqobj);
                    sprintf(errmsg, "Invalid custom axes is_modulo value (not True or False) for the %s axis", AXIS_NAMES[k]);
                    return;
                }
            }
            /* Make the assignment for this axis */
            Py_DECREF(subseqobj);
            q = k+1;
            ef_set_custom_axis_sub_(&id, &q, &(values[0]), &(values[1]), &(values[2]), unit_name, &is_modulo);
            call_made = 1;
        }
    }
    Py_DECREF(seqobj);

    /* Make sure ef_set_custom_axis_sub_ was called at least once */
    if ( ! call_made )
        sprintf(errmsg, "No custom axis value were given in the tuple returned from %s in %s", CUSTOM_AXES_METHOD_NAME, modname);
    else
        errmsg[0] = '\0';
    return;
}

