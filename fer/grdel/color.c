/*
 * Color objects can only be used with the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "pyferret.h"

static const char *grdelcolorid = "GRDEL_COLOR";

typedef struct GDColor_ {
    const char *id;
    grdelType window;
    PyObject *object;
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
    PyObject *bindings;
    GDColor *color;

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "grdelColor: window argument is not "
                            "a grdel Window");
        return NULL;
    }

    if ( (0.0 > redfrac) || (redfrac > 1.0) ) {
        strcpy(grdelerrmsg, "grdelColor: redfrac must be in [0.0, 1.0]");
        return NULL;
    }
    if ( (0.0 > greenfrac) || (greenfrac > 1.0) ) {
        strcpy(grdelerrmsg, "grdelColor: greenfrac must be in [0.0, 1.0]");
        return NULL;
    }
    if ( (0.0 > bluefrac) || (bluefrac > 1.0) ) {
        strcpy(grdelerrmsg, "grdelColor: bluefrac must be in [0.0, 1.0]");
        return NULL;
    }
    if ( (0.0 > opaquefrac) || (opaquefrac > 1.0) ) {
        strcpy(grdelerrmsg, "grdelColor: opaquefrac must be in [0.0, 1.0]");
        return NULL;
    }

    color = (GDColor *) PyMem_Malloc(sizeof(GDColor));
    if ( color == NULL ) {
        strcpy(grdelerrmsg, "grdelColor: out of memory for a new Color");
        return NULL;
    }

    color->id = grdelcolorid;
    color->window = window;
    color->object = PyObject_CallMethod(bindings, "createColor", "dddd",
                                 (double) redfrac, (double) greenfrac,
                                 (double) bluefrac, (double) opaquefrac);
    if ( color->object == NULL ) {
        sprintf(grdelerrmsg, "grdelColor: error when calling the "
                "binding's createColor method: %s", pyefcn_get_error());
        PyMem_Free(color);
        return NULL;
    }

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelColor created: "
            "window = %X, rgba = (%f,%f,%f,%f), color = %X\n",
            window, redfrac, greenfrac, bluefrac, opaquefrac, color);
    fflush(debuglogfile);
#endif

    grdelerrmsg[0] = '\0';
    return color;
}

/*
 * Verifies color is a grdel Color.  If window is not NULL,
 * also verifies color can be used with this Window.
 * Returns a pointer to the graphic engine's color object
 * if successful.  Returns NULL if there is a problem.
 */
PyObject *grdelColorVerify(grdelType color, grdelType window)
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
    GDColor *mycolor;
    PyObject *bindings;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelColorDelete called: "
            "color = %X\n", color);
    fflush(debuglogfile);
#endif

    if ( grdelColorVerify(color, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelColorDelete: color argument is not "
                            "a grdel Color");
        return (grdelBool) 0;
    }
    mycolor = (GDColor *) color;

    bindings = grdelWindowVerify(mycolor->window);
    /* "N" - steals the reference to this color object */
    result = PyObject_CallMethod(bindings, "deleteColor", "N",
                                           mycolor->object);
    if ( result == NULL )
        sprintf(grdelerrmsg, "grdelColorDelete: error when calling the "
                "binding's deleteColor method: %s", pyefcn_get_error());
    else
        Py_DECREF(result);

    mycolor->id = NULL;
    mycolor->window = NULL;
    mycolor->object = NULL;
    PyMem_Free(color);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
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
void fgdcolor_(void **color, void **window, float *redfrac,
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
void fgdcolordel_(int *success, void **color)
{
    grdelBool result;

    result = grdelColorDelete(*color);
    *success = result;
}

