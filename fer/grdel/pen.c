/*
 * Pen objects can only be used with the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "pyferret.h"
#include "FerMem.h"

static const char *grdelpenid = "GRDEL_PEN";

typedef struct GDpen_ {
    const char *id;
    grdelType window;
    grdelType object;
} GDPen;


/*
 * Returns a Pen object.
 *
 * Arguments:
 *     window: Window in which this pen is to be used
 *     color: Color to use
 *     width: line width in units of 0.001 of the length
 *            of the longest side of the view
 *     style: line style name (e.g., "solid", "dash")
 *     stylelen: actual length of the style name
 *     capstyle: end-cap style name (e.g., "square")
 *     capstylelen: actual length of the capstyle name
 *     joinstyle: join style name (e.g., "bevel")
 *     joinstylelen: actual length of the joinstyle name
 *
 * Returns a pointer to the pen object created.  If an error occurs,
 * NULL is returned and grdelerrmsg contains an explanatory message.
 */
grdelType grdelPen(grdelType window, grdelType color,
                   float width, const char *style, int stylelen,
                   const char *capstyle, int capstylelen,
                   const char *joinstyle, int joinstylelen)
{
    const BindObj *bindings;
    grdelType colorobj;
    GDPen    *pen;

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "grdelPen: window argument is not "
                            "a grdel Window");
        return NULL;
    }
    colorobj = grdelColorVerify(color, window);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "grdelPen: color argument is not "
                            "a valid grdel Color for the window");
        return NULL;
    }

    pen = (GDPen *) FerMem_Malloc(sizeof(GDPen), __FILE__, __LINE__);
    if ( pen == NULL ) {
        strcpy(grdelerrmsg, "grdelPen: out of memory for a new Pen");
        return NULL;
    }

    pen->id = grdelpenid;
    pen->window = window;
    if ( bindings->cferbind != NULL ) {
        pen->object = bindings->cferbind->createPen(bindings->cferbind,
                                colorobj, (double) width, style, stylelen,
                                capstyle, capstylelen, joinstyle, joinstylelen);
        if ( pen->object == NULL ) {
            /* grdelerrmsg already assigned */
            FerMem_Free(pen, __FILE__, __LINE__);
            return NULL;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        pen->object = PyObject_CallMethod(bindings->pyobject, "createPen",
                               "Ods#s#s#", (PyObject *) colorobj,
                               (double) width, style, stylelen, capstyle,
                               capstylelen, joinstyle, joinstylelen);
        if ( pen->object == NULL ) {
            sprintf(grdelerrmsg, "grdelPen: error when calling the Python "
                    "binding's createPen method: %s", pyefcn_get_error());
            FerMem_Free(pen, __FILE__, __LINE__);
            return NULL;
        }
    }
    else {
        strcpy(grdelerrmsg, "grdelPen: unexpected error, "
                            "no bindings associated with this Window");
        FerMem_Free(pen, __FILE__, __LINE__);
        return NULL;
    }

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelPen created: "
            "window = %p, color = %p, width = %f, pen = %p\n",
            window, color, width, pen);
    fflush(debuglogfile);
#endif

    return pen;
}

/*
 * Verifies pen is a grdel Pen.  If window is not NULL,
 * also verifies pen can be used with this Window.
 * Returns a pointer to the graphic engine's pen object
 * if successful.  Returns NULL if there is a problem.
 */
grdelType grdelPenVerify(grdelType pen, grdelType window)
{
    GDPen *mypen;

    if ( pen == NULL )
        return NULL;
    mypen = (GDPen *) pen;
    if ( mypen->id != grdelpenid )
        return NULL;
    if ( (window != NULL) && (mypen->window != window) )
        return NULL;
    return mypen->object;
}

/*
 * Replace the color in the given pen object with that in
 * the given color object.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool grdelPenReplaceColor(grdelType pen, grdelType color)
{
    const BindObj *bindings;
    GDPen     *mypen;
    grdelType *colorobj;
    grdelBool  success;
    PyObject  *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelPenReplaceColor called: "
            "pen = %p, color = %p\n", pen, color);
    fflush(debuglogfile);
#endif

    if ( grdelPenVerify(pen, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelPenReplaceColor: pen argument is not "
                            "a grdel Pen");
        return 0;
    }
    mypen = (GDPen *) pen;

    colorobj = grdelColorVerify(color, mypen->window);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "grdelPenReplaceColor: color argument is not "
                            "a valid grdel Color for the window");
        return 0;
    }

    success = 1;

    bindings = grdelWindowVerify(mypen->window);
    if ( bindings->cferbind != NULL ) {
        success = bindings->cferbind->replacePenColor(bindings->cferbind,
                                                mypen->object, colorobj);
        /* if there was a problem, grdelerrmsg is already assigned */
    }
    else if ( bindings->pyobject != NULL ) {
        result = PyObject_CallMethod(bindings->pyobject, "replacePenColor",
                                     "OO", (PyObject *) mypen->object,
                                           (PyObject *) colorobj);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelPenDelete: error when calling the "
                    "Python binding's replacePenColor method: %s", pyefcn_get_error());
            success = 0;
        }
        else
            Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelPenReplaceColor: unexpected error, "
                            "no bindings associated with this Window");
        success = 0;
    }

    return success;
}

/*
 * Delete a Pen created by grdelPen
 *
 * Arguments:
 *     pen: Pen to be deleted
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelPenDelete(grdelType pen)
{
    const BindObj *bindings;
    GDPen    *mypen;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelPenDelete called: "
            "pen = %p\n", pen);
    fflush(debuglogfile);
#endif

    if ( grdelPenVerify(pen, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelPenDelete: pen argument is not "
                            "a grdel Pen");
        return 0;
    }
    mypen = (GDPen *) pen;

    success = 1;

    bindings = grdelWindowVerify(mypen->window);
    if ( bindings->cferbind != NULL ) {
        success = bindings->cferbind->deletePen(bindings->cferbind,
                                                mypen->object);
        /* if there was a problem, grdelerrmsg is already assigned */
    }
    else if ( bindings->pyobject != NULL ) {
        /* "N" - steals the reference to this pen object */
        result = PyObject_CallMethod(bindings->pyobject, "deletePen",
                                     "N", (PyObject *) mypen->object);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelPenDelete: error when calling the Python "
                    "binding's deletePen method: %s", pyefcn_get_error());
            success = 0;
        }
        else
            Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelPenDelete: unexpected error, "
                            "no bindings associated with this Window");
        success = 0;
    }

    /* regardless of success, free this Pen */
    mypen->id = NULL;
    mypen->window = NULL;
    mypen->object = NULL;
    FerMem_Free(mypen, __FILE__, __LINE__);

    return success;
}

/*
 * Creates a Pen object.
 *
 * Input Arguments:
 *     window: Window in which this pen is to be used
 *     color: Color to use
 *     width: line width in units of 0.001 of the length
 *            of the longest side of the view
 *     style: line style name (e.g., "solid", "dash")
 *     stylelen: actual length of the style name
 *     capstyle: end-cap style name (e.g., "square")
 *     capstylelen: actual length of the capstyle name
 *     joinstyle: join style name (e.g., "bevel")
 *     joinstylelen: actual length of the joinstyle name
 * Output Arguments:
 *     pen: the created pen object, or zero if failure.
 *             Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdpen)(void **pen, void **window, void **color, float *width,
             char *style, int *stylelen, char *capstyle, int *capstylelen,
             char *joinstyle, int *joinstylelen)
{
    grdelType mypen;

    mypen = grdelPen(*window, *color, *width, style, *stylelen,
                     capstyle, *capstylelen, joinstyle, *joinstylelen);
    *pen = mypen;
}

/*
 * Replace a color used in a pen
 *
 * Input Arguments:
 *     pen: Pen to be modified
 *     color: new Color to use in pen
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdpenreplacecolor)(int *success, void **pen, void **color)
{
    grdelBool result;

    result = grdelPenReplaceColor(*pen, *color);
    *success = result;
}

/*
 * Deletes a Pen created by fgdpen_
 *
 * Input Arguments:
 *     pen: Pen to be deleted
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdpendel)(int *success, void **pen)
{
    grdelBool result;

    result = grdelPenDelete(*pen);
    *success = result;
}

