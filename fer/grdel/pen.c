/*
 * Pen objects can only be used with the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "pyferret.h"

static const char *grdelpenid = "GRDEL_PEN";

typedef struct GDpen_ {
    const char *id;
    grdelType window;
    PyObject *object;
} GDPen;


/*
 * Returns a Pen object.
 *
 * Arguments:
 *     window: Window in which this pen is to be used
 *     color: Color to use
 *     width: line width in View units
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
    PyObject *bindings;
    PyObject *colorobj;
    GDPen *pen;

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

    pen = (GDPen *) PyMem_Malloc(sizeof(GDPen));
    if ( pen == NULL ) {
        strcpy(grdelerrmsg, "grdelPen: out of memory for a new Pen");
        return NULL;
    }

    pen->id = grdelpenid;
    pen->window = window;
    pen->object = PyObject_CallMethod(bindings, "createPen", "Ods#s#s#",
                           colorobj, (double) width, style, stylelen,
                           capstyle, capstylelen, joinstyle, joinstylelen);
    if ( pen->object == NULL ) {
        sprintf(grdelerrmsg, "grdelPen: error when calling "
                "the binding's createPen method: %s", pyefcn_get_error());
        PyMem_Free(pen);
        return NULL;
    }

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelPen created: "
            "window = %X, color = %X, width = %f, pen = %X\n",
            window, color, width, pen);
    fflush(debuglogfile);
#endif

    grdelerrmsg[0] = '\0';
    return pen;
}

/*
 * Verifies pen is a grdel Pen.  If window is not NULL,
 * also verifies pen can be used with this Window.
 * Returns a pointer to the graphic engine's pen object
 * if successful.  Returns NULL if there is a problem.
 */
PyObject *grdelPenVerify(grdelType pen, grdelType window)
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
    GDPen *mypen;
    PyObject *bindings;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelPenDelete called: "
            "pen = %X\n", pen);
    fflush(debuglogfile);
#endif

    if ( grdelPenVerify(pen, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelPenDelete: pen argument is not "
                            "a grdel Pen");
        return (grdelBool) 0;
    }
    mypen = (GDPen *) pen;

    bindings = grdelWindowVerify(mypen->window);
    /* "N" - steals the reference to this pen object */
    result = PyObject_CallMethod(bindings, "deletePen", "N",
                                 mypen->object);
    if ( result == NULL )
        sprintf(grdelerrmsg, "grdelPenDelete: error when calling "
                "the binding's deletePen method: %s", pyefcn_get_error());
    else
        Py_DECREF(result);

    mypen->id = NULL;
    mypen->window = NULL;
    mypen->object = NULL;
    PyMem_Free(mypen);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Creates a Pen object.
 *
 * Input Arguments:
 *     window: Window in which this pen is to be used
 *     color: Color to use
 *     width: line width in View units
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
void fgdpen_(void **pen, void **window, void **color, float *width,
             char *style, int *stylelen, char *capstyle, int *capstylelen,
             char *joinstyle, int *joinstylelen)
{
    grdelType mypen;

    mypen = grdelPen(*window, *color, *width, style, *stylelen,
                     capstyle, *capstylelen, joinstyle, *joinstylelen);
    *pen = mypen;
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
void fgdpendel_(int *success, void **pen)
{
    grdelBool result;

    result = grdelPenDelete(*pen);
    *success = result;
}

