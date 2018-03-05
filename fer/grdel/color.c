/*
 * Color objects can only be used with the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "pyferret.h"
#include "FerMem.h"

static const char *grdelcolorid = "GRDEL_COLOR";

typedef struct GDColor_ {
    const char *id;
    grdelType window;
    grdelType object;
} GDColor;


/*
 * Returns a Color object from fractional [0.0, 1.0] intensities
 * of the red, green, and blue channels.
 *
 * Arguments:
 *     window: Window in which this color is to be used
 *     redfrac: fractional [0.0, 1.0] red intensity
 *     greenfrac: fractional [0.0, 1.0] green intensity
 *     bluefrac: fractional [0.0, 1.0] blue intensity
 *     opaquefrac: fractional [0.0, 1.0] opaqueness
 *             (0.0 is transparent; 1.0 is opaque) of the color.
 *             If the graphics engine does not support this
 *             feature (alpha channel), this may be silently
 *             ignored and the color be completely opaque.
 *
 * Returns a pointer to the color object created.  If an error occurs,
 * NULL is returned and grdelerrmsg contains an explanatory message.
 */
grdelType grdelColor(grdelType window, float redfrac, float greenfrac,
                                       float bluefrac, float opaquefrac)
{
    const BindObj *bindings;
    GDColor *color;

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "grdelColor: window argument is not "
                            "a grdel Window");
        return NULL;
    }

    if ( (0.0 > redfrac) || (redfrac > 1.0) ) {
        sprintf(grdelerrmsg, "grdelColor: redfrac (%.2f) must be in [0.0, 1.0]", redfrac);
        return NULL;
    }
    if ( (0.0 > greenfrac) || (greenfrac > 1.0) ) {
        sprintf(grdelerrmsg, "grdelColor: greenfrac (%.2f) must be in [0.0, 1.0]", greenfrac);
        return NULL;
    }
    if ( (0.0 > bluefrac) || (bluefrac > 1.0) ) {
        sprintf(grdelerrmsg, "grdelColor: bluefrac (%.2f) must be in [0.0, 1.0]", bluefrac);
        return NULL;
    }
    if ( (0.0 > opaquefrac) || (opaquefrac > 1.0) ) {
        sprintf(grdelerrmsg, "grdelColor: opaquefrac (%.2f) must be in [0.0, 1.0]", opaquefrac);
        return NULL;
    }

    color = (GDColor *) FerMem_Malloc(sizeof(GDColor), __FILE__, __LINE__);
    if ( color == NULL ) {
        strcpy(grdelerrmsg, "grdelColor: out of memory for a new Color");
        return NULL;
    }

    color->id = grdelcolorid;
    color->window = window;
    if ( bindings->cferbind != NULL ) {
        color->object = bindings->cferbind->createColor(bindings->cferbind,
                                  (double) redfrac, (double) greenfrac,
                                  (double) bluefrac, (double) opaquefrac);
        if ( color->object == NULL ) {
            /* grdelerrmsg already assigned */
            FerMem_Free(color, __FILE__, __LINE__);
            return NULL;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        color->object = PyObject_CallMethod(bindings->pyobject, "createColor",
                                 "dddd", (double) redfrac, (double) greenfrac,
                                 (double) bluefrac, (double) opaquefrac);
        if ( color->object == NULL ) {
            sprintf(grdelerrmsg, "grdelColor: error when calling the Python "
                    "binding's createColor method: %s", pyefcn_get_error());
            FerMem_Free(color, __FILE__, __LINE__);
            return NULL;
        }
    }
    else {
        strcpy(grdelerrmsg, "grdelColor: unexpected error, "
                            "no bindings associated with this Window");
        FerMem_Free(color, __FILE__, __LINE__);
        return NULL;
    }

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelColor created: "
            "window = %p, rgba = (%f,%f,%f,%f), color = %p\n",
            window, redfrac, greenfrac, bluefrac, opaquefrac, color);
    fflush(debuglogfile);
#endif

    return color;
}

/*
 * Verifies color is a grdel Color.  If window is not NULL,
 * also verifies color can be used with this Window.
 * Returns a pointer to the graphic engine's color object
 * if successful.  Returns NULL if there is a problem.
 */
grdelType grdelColorVerify(grdelType color, grdelType window)
{
    GDColor *mycolor;

    if ( color == NULL )
        return NULL;
    mycolor = (GDColor *) color;
    if ( mycolor->id != grdelcolorid )
        return NULL;
    if ( (window != NULL) && (mycolor->window != window) )
        return NULL;
    return mycolor->object;
}

/*
 * Delete a Color created by grdelColor
 *
 * Arguments:
 *     color: Color to be deleted
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelColorDelete(grdelType color)
{
    const BindObj *bindings;
    GDColor  *mycolor;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelColorDelete called: "
            "color = %p\n", color);
    fflush(debuglogfile);
#endif

    if ( grdelColorVerify(color, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelColorDelete: color argument is not "
                            "a grdel Color");
        return 0;
    }
    mycolor = (GDColor *) color;

    success = 1;

    bindings = grdelWindowVerify(mycolor->window);
    if ( bindings->cferbind != NULL ) {
        success = bindings->cferbind->deleteColor(bindings->cferbind,
                                                  mycolor->object);
        /* if there was a problem, grdelerrmsg is already assigned */
    }
    else if ( bindings->pyobject != NULL ) {
        /* "N" - steals the reference to this color object */
        result = PyObject_CallMethod(bindings->pyobject, "deleteColor",
                                     "N", (PyObject *) mycolor->object);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelColorDelete: error when calling the "
                    "Python binding's deleteColor method: %s", pyefcn_get_error());
            success = 0;
        }
        else
            Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelColorDelete: unexpected error, "
                            "no bindings associated with this Window");
        success = 0;
    }

    /* regardless of success, free this Color */
    mycolor->id = NULL;
    mycolor->window = NULL;
    mycolor->object = NULL;
    FerMem_Free(color, __FILE__, __LINE__);

    return success;
}

/*
 * Creates a Color object from fractional [0.0, 1.0] intensities
 * of the red, green, and blue channels.
 *
 * Input Arguments:
 *     window: Window in which this color is to be used
 *     redfrac: fractional [0.0, 1.0] red intensity
 *     greenfrac: fractional [0.0, 1.0] green intensity
 *     bluefrac: fractional [0.0, 1.0] blue intensity
 *     opaquefrac: fractional [0.0, 1.0] opaqueness
 *             (0.0 is transparent; 1.0 is opaque) of the color.
 *             If the graphics engine does not support this
 *             feature (alpha channel), this may be silently
 *             ignored and the color be completely opaque.
 * Output Arguments:
 *     color: the created color object, or zero if failure.
 *             Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdcolor)(void **color, void **window, float *redfrac,
               float *greenfrac, float *bluefrac, float *opaquefrac)
{
    grdelType mycolor;

    mycolor = grdelColor(*window, *redfrac, *greenfrac,
                         *bluefrac, *opaquefrac);
    *color = mycolor;
}

/*
 * Deletes a Color created by fgdcolor_
 *
 * Input Arguments:
 *     color: Color to be deleted
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdcolordel)(int *success, void **color)
{
    grdelBool result;

    result = grdelColorDelete(*color);
    *success = result;
}

