/*
 * Brush objects can only be used with the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "pyferret.h"

static const char *grdelbrushid = "GRDEL_BRUSH";

typedef struct GDbrush_ {
    const char *id;
    grdelType window;
    PyObject *object;
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
    PyObject *bindings;
    PyObject *colorobj;
    GDBrush *brush;

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
    brush->object = PyObject_CallMethod(bindings, "createBrush", "Os#",
                                        colorobj, style, stylelen);
    if ( brush->object == NULL ) {
        sprintf(grdelerrmsg, "grdelBrush: error when calling the "
                "binding's createBrush method: %s", pyefcn_get_error());
        PyMem_Free(brush);
        return NULL;
    }

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelBrush created: "
            "window = %X, color = %X, brush = %X\n",
            window, color, brush);
    fflush(debuglogfile);
#endif

    grdelerrmsg[0] = '\0';
    return brush;
}

/*
 * Verifies brush is a grdel Brush.  If window is not NULL,
 * also verifies brush can be used with this Window.
 * Returns a pointer to the graphic engine's brush object
 * if successful.  Returns NULL if there is a problem.
 */
PyObject *grdelBrushVerify(grdelType brush, grdelType window)
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
    GDBrush *mybrush;
    PyObject *bindings;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelBrushDelete called: "
            "brush = %X\n", brush);
    fflush(debuglogfile);
#endif

    if ( grdelBrushVerify(brush, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelBrushDelete: brush argument is not "
                            "a grdel Brush");
        return (grdelBool) 0;
    }
    mybrush = (GDBrush *) brush;

    bindings = grdelWindowVerify(mybrush->window);
    /* "N" - steals the reference to this brush object */
    result = PyObject_CallMethod(bindings, "deleteBrush", "N",
                                           mybrush->object);
    if ( result == NULL )
        sprintf(grdelerrmsg, "grdelBrushDelete: error when calling the "
                "binding's deleteBrush method: %s", pyefcn_get_error());
    else
        Py_DECREF(result);

    mybrush->id = NULL;
    mybrush->window = NULL;
    mybrush->object = NULL;
    PyMem_Free(brush);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
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

