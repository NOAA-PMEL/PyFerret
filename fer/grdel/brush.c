/*
 * Brush objects can only be used with the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "pyferret.h"
#include "FerMem.h"

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

    brush = (GDBrush *) FerMem_Malloc(sizeof(GDBrush), __FILE__, __LINE__);
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
            FerMem_Free(brush, __FILE__, __LINE__);
            return NULL;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        brush->object = PyObject_CallMethod(bindings->pyobject, "createBrush",
                                 "Os#", (PyObject *) colorobj, style, stylelen);
        if ( brush->object == NULL ) {
            sprintf(grdelerrmsg, "grdelBrush: error when calling the Python "
                    "binding's createBrush method: %s", pyefcn_get_error());
            FerMem_Free(brush, __FILE__, __LINE__);
            return NULL;
        }
    }
    else {
        strcpy(grdelerrmsg, "grdelBrush: unexpected error, "
                            "no bindings associated with this Window");
        FerMem_Free(brush, __FILE__, __LINE__);
        return NULL;
    }

#ifdef GRDELDEBUG
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
 * Replace the color in the given brush object with that in
 * the given color object.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool grdelBrushReplaceColor(grdelType brush, grdelType color)
{
    const BindObj *bindings;
    GDBrush   *mybrush;
    grdelType *colorobj;
    grdelBool  success;
    PyObject  *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelBrushReplaceColor called: "
            "brush = %p, color = %p\n", brush, color);
    fflush(debuglogfile);
#endif

    if ( grdelBrushVerify(brush, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelBrushReplaceColor: brush argument is not "
                            "a grdel Brush");
        return 0;
    }
    mybrush = (GDBrush *) brush;

    colorobj = grdelColorVerify(color, mybrush->window);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "grdelBrushReplaceColor: color argument is not "
                            "a valid grdel Color for the window");
        return 0;
    }

    success = 1;

    bindings = grdelWindowVerify(mybrush->window);
    if ( bindings->cferbind != NULL ) {
        success = bindings->cferbind->replaceBrushColor(bindings->cferbind,
                                                  mybrush->object, colorobj);
        /* if there was a problem, grdelerrmsg is already assigned */
    }
    else if ( bindings->pyobject != NULL ) {
        result = PyObject_CallMethod(bindings->pyobject, "replaceBrushColor",
                                     "OO", (PyObject *) mybrush->object,
                                           (PyObject *) colorobj);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelBrushDelete: error when calling the "
                    "Python binding's replaceBrushColor method: %s", pyefcn_get_error());
            success = 0;
        }
        else
            Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelBrushReplaceColor: unexpected error, "
                            "no bindings associated with this Window");
        success = 0;
    }

    return success;
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

#ifdef GRDELDEBUG
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
    FerMem_Free(brush, __FILE__, __LINE__);

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
 * Replace a color used in a brush
 *
 * Input Arguments:
 *     brush: Brush to be modified
 *     color: new Color to use in brush
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdbrushreplacecolor_(int *success, void **brush, void **color)
{
    grdelBool result;

    result = grdelBrushReplaceColor(*brush, *color);
    *success = result;
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

