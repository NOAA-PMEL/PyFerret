/*
 * Brush objects can only be used with the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "pyferret.h"

static const char *grdelbrushid = "GRDEL_BRUSH";

typedef struct GDbrush_ {
    const char *id;
    grdelType window;
    grdelType object;
} GDBrush;


/*
 * Returns a Brush object.
 *
 * Arguments:
 *     window: Window in which this brush is to be used
 *     color: Color to use
 *     style: fill style name (e.g., "solid", "cross")
 *     stylelen: actual length of the style name
 *
 * Returns a pointer to the brush object created.  If an error occurs,
 * NULL is returned and grdelerrmsg contains an explanatory message.
 */
grdelType grdelBrush(grdelType window, grdelType color,
                     const char *style, int stylelen)
{
    const BindObj *bindings;
    grdelType *colorobj;
    GDBrush   *brush;

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "grdelBrush: window argument is not "
                            "a grdel Window");
        return NULL;
    }
    colorobj = grdelColorVerify(color, window);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "grdelBrush: color argument is not "
                            "a valid grdel Color for the window");
        return NULL;
    }

    brush = (GDBrush *) PyMem_Malloc(sizeof(GDBrush));
    if ( brush == NULL ) {
        strcpy(grdelerrmsg, "grdelBrush: out of memory for a new Brush");
        return NULL;
    }

    brush->id = grdelbrushid;
    brush->window = window;
    if ( bindings->cferbind != NULL ) {
        brush->object = bindings->cferbind->createBrush(bindings->cferbind,
                                                  colorobj, style, stylelen);
        if ( brush->object == NULL ) {
            /* grdelerrmsg already assigned */
            PyMem_Free(brush);
            return NULL;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        brush->object = PyObject_CallMethod(bindings->pyobject, "createBrush",
                                 "Os#", (PyObject *) colorobj, style, stylelen);
        if ( brush->object == NULL ) {
            sprintf(grdelerrmsg, "grdelBrush: error when calling the Python "
                    "binding's createBrush method: %s", pyefcn_get_error());
            PyMem_Free(brush);
            return NULL;
        }
    }
    else {
        strcpy(grdelerrmsg, "grdelBrush: unexpected error, "
                            "no bindings associated with this Window");
        PyMem_Free(brush);
        return NULL;
    }

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelBrush created: "
            "window = %p, color = %p, brush = %p\n",
            window, color, brush);
    fflush(debuglogfile);
#endif

    return brush;
}

/*
 * Verifies brush is a grdel Brush.  If window is not NULL,
 * also verifies brush can be used with this Window.
 * Returns a pointer to the graphic engine's brush object
 * if successful.  Returns NULL if there is a problem.
 */
grdelType grdelBrushVerify(grdelType brush, grdelType window)
{
    GDBrush *mybrush;

    if ( brush == NULL )
        return NULL;
    mybrush = (GDBrush *) brush;
    if ( mybrush->id != grdelbrushid )
        return NULL;
    if ( (window != NULL) && (mybrush->window != window) )
        return NULL;
    return mybrush->object;
}

/*
 * Delete a Brush created by grdelBrush
 *
 * Arguments:
 *     brush: Brush to be deleted
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelBrushDelete(grdelType brush)
{
    const BindObj *bindings;
    GDBrush  *mybrush;
    grdelBool success;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelBrushDelete called: "
            "brush = %p\n", brush);
    fflush(debuglogfile);
#endif

    if ( grdelBrushVerify(brush, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelBrushDelete: brush argument is not "
                            "a grdel Brush");
        return 0;
    }
    mybrush = (GDBrush *) brush;

    success = 1;

    bindings = grdelWindowVerify(mybrush->window);
    if ( bindings->cferbind != NULL ) {
        success = bindings->cferbind->deleteBrush(bindings->cferbind,
                                                  mybrush->object);
        /* if there was a problem, grdelerrmsg is already assigned */
    }
    else if ( bindings->pyobject != NULL ) {
        /* "N" - steals the reference to this brush object */
        result = PyObject_CallMethod(bindings->pyobject, "deleteBrush",
                                     "N", (PyObject *) mybrush->object);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelBrushDelete: error when calling the "
                    "Python binding's deleteBrush method: %s", pyefcn_get_error());
            success = 0;
        }
        else
            Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelBrushDelete: unexpected error, "
                            "no bindings associated with this Window");
        success = 0;
    }

    /* regardless of success, free this Brush */
    mybrush->id = NULL;
    mybrush->window = NULL;
    mybrush->object = NULL;
    PyMem_Free(brush);

    return success;
}

/*
 * Creates a Brush object.
 *
 * Input Arguments:
 *     window: Window in which this brush is to be used
 *     color: Color to use
 *     style: fill style name (e.g., "solid", "cross")
 *     stylelen: actual length of the style name
 * Output Arguments:
 *     brush: the created brush object, or zero if failure.
 *             Use fgderrmsg_ to retrieve the error message.
 */
void fgdbrush_(void **brush, void **window, void **color,
               char *style, int *stylelen)
{
    grdelType mybrush;

    mybrush = grdelBrush(*window, *color, style, *stylelen);
    *brush = mybrush;
}

/*
 * Deletes a Brush created by fgdbrush_
 *
 * Input Arguments:
 *     brush: Brush to be deleted
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdbrushdel_(int *success, void **brush)
{
    grdelBool result;

    result = grdelBrushDelete(*brush);
    *success = result;
}

