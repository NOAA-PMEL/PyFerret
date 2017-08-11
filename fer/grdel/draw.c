/*
 * Drawing commands
 */
#include <Python.h> /* make sure Python.h is first */
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "pyferret.h"
#include "FerMem.h"

/*
 * Draws connected line segments.
 *
 * Arguments:
 *     window: Window with an active View to draw in
 *     ptsx: user X-coordinates of the endpoint
 *     ptsy: user Y-coordinates of the endpoints
 *     numpts: number of coordinates in ptsx and ptsy to use
 *     pen: Pen to use to draw the line segments
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawMultiline(grdelType window, const float ptsx[],
               const float ptsy[], int numpts, grdelType pen)
{
    const BindObj *bindings;
    grdelType penobj;
    double   *xvals;
    double   *yvals;
    grdelBool success;
    PyObject *xtuple;
    PyObject *ytuple;
    PyObject *fltobj;
    PyObject *result;
    double my, sx, sy, dx, dy;
    double transval;
    int k;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelDrawMultiline called: "
            "window = %p, pen = %p, numpts = %d\n", window, pen, numpts);
    for (k = 0; k < numpts; k++)
        fprintf(debuglogfile, "   pt[%d] = (%f,%f)\n", k, ptsx[k], ptsy[k]);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawMultiline: window argument is not "
                            "a grdel Window");
        return 0;
    }
    penobj = grdelPenVerify(pen, window);
    if ( penobj == NULL ) {
        strcpy(grdelerrmsg, "grdelDrawMultiline: pen argument is not "
                            "a valid grdel Pen for the window");
        return 0;
    }
    if ( numpts <= 1 ) {
        strcpy(grdelerrmsg, "grdelDrawMultiline: invalid number of points");
        return 0;
    }

    /* Get the transform values for converting user to device coordinates */
    grdelGetTransformValues(&my, &sx, &sy, &dx, &dy);

    if ( bindings->cferbind != NULL ) {
        xvals = (double *) FerMem_Malloc(2 * numpts * sizeof(double), __FILE__, __LINE__);
        if ( xvals == NULL ) {
            sprintf(grdelerrmsg, "grdelDrawMultiline: out of memory "
                                 "for an array of %d doubles", 2 * numpts);
            return 0;
        }
        yvals = &(xvals[numpts]);
        for (k = 0; k < numpts; k++)
            xvals[k] = (double) (ptsx[k]) * sx + dx;
        for (k = 0; k < numpts; k++)
            yvals[k] = (my - (double) (ptsy[k])) * sy + dy;
        success = bindings->cferbind->drawMultiline(bindings->cferbind,
                                      xvals, yvals, numpts, penobj);
        FerMem_Free(xvals, __FILE__, __LINE__);
        if ( ! success ) {
            /* grdelerrmsg is already assigned */
            return 0;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        xtuple = PyTuple_New( (Py_ssize_t) numpts );
        if ( xtuple == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawMultiline: problems creating "
                                "a Python tuple");
            return 0;
        }
        for (k = 0; k < numpts; k++) {
            transval = (double) (ptsx[k]) * sx + dx;
            fltobj = PyFloat_FromDouble(transval);
            if ( fltobj == NULL ) {
                PyErr_Clear();
                strcpy(grdelerrmsg, "grdelDrawMultiline: problems creating "
                                    "a Python float");
                Py_DECREF(xtuple);
                return 0;
            }
            /* PyTuple_SET_ITEM steals the reference to fltobj */
            PyTuple_SET_ITEM(xtuple, (Py_ssize_t) k, fltobj);
        }

        ytuple = PyTuple_New( (Py_ssize_t) numpts );
        if ( ytuple == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawMultiline: problems creating "
                                "a Python tuple");
            Py_DECREF(xtuple);
            return 0;
        }
        for (k = 0; k < numpts; k++) {
            transval = (my - (double) (ptsy[k])) * sy + dy;
            fltobj = PyFloat_FromDouble(transval);
            if ( fltobj == NULL ) {
                PyErr_Clear();
                strcpy(grdelerrmsg, "grdelDrawMultiline: problems creating "
                                    "a Python float");
                Py_DECREF(ytuple);
                Py_DECREF(xtuple);
                return 0;
            }
            /* PyTuple_SET_ITEM steals the reference to fltobj */
            PyTuple_SET_ITEM(ytuple, (Py_ssize_t) k, fltobj);
        }

        /*
         * Call the drawMultiline method of the bindings instance.
         * Using 'N' to steal the reference to xtuple and to ytuple.
         */
        result = PyObject_CallMethod(bindings->pyobject, "drawMultiline",
                          "NNO", xtuple, ytuple, (PyObject *) penobj);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelDrawMultiline: error when calling the Python "
                    "binding's drawMultiline method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdeldrawMultiline: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Draws discrete points.
 *
 * Arguments:
 *     window: Window with an active View to draw in
 *     ptsx: user X-coordinates of the points
 *     ptsy: user Y-coordinates of the points
 *     numpts: number of coordinates in ptsx and ptsy to use
 *     symbol: Symbol to use to draw a point
 *     color: color of the Symbol
 *     ptsize: size of the symbol (scales with view size)
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawPoints(grdelType window, const float ptsx[],
               const float ptsy[], int numpts, grdelType symbol,
               grdelType color, float ptsize)
{
    const BindObj *bindings;
    grdelType symbolobj;
    grdelType colorobj;
    double   *xvals;
    double   *yvals;
    grdelBool success;
    PyObject *xtuple;
    PyObject *ytuple;
    PyObject *fltobj;
    PyObject *result;
    double my, sx, sy, dx, dy;
    double transval;
    int k;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelDrawPoints called: "
            "window = %p, symbol = %p, color = %p, ptsize = %f, numpts = %d", 
            window, symbol, color, ptsize, numpts);
    for (k = 0; k < numpts; k++)
        fprintf(debuglogfile, "   pt[%d] = (%f,%f)\n", k, ptsx[k], ptsy[k]);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawPoints: window argument is not "
                            "a grdel Window");
        return 0;
    }
    symbolobj = grdelSymbolVerify(symbol, window);
    if ( symbolobj == NULL ) {
        strcpy(grdelerrmsg, "grdelDrawPoints: symbol argument is not "
                            "a valid grdel Symbol for the window");
        return 0;
    }
    colorobj = grdelColorVerify(color, window);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "grdelDrawPoints: color argument is not "
                            "a valid grdel Color for the window");
        return 0;
    }
    if ( numpts <= 0 ) {
        strcpy(grdelerrmsg, "grdelDrawPoints: invalid number of points");
        return 0;
    }

    /* Get the transform values for converting user to device coordinates */
    grdelGetTransformValues(&my, &sx, &sy, &dx, &dy);

    if ( bindings->cferbind != NULL ) {
        xvals = (double *) FerMem_Malloc(2 * numpts * sizeof(double), __FILE__, __LINE__);
        if ( xvals == NULL ) {
            sprintf(grdelerrmsg, "grdelDrawPoints: out of memory "
                                 "for an array of %d doubles", 2 * numpts);
            return 0;
        }
        yvals = &(xvals[numpts]);
        for (k = 0; k < numpts; k++)
            xvals[k] = (double) (ptsx[k]) * sx + dx;
        for (k = 0; k < numpts; k++)
            yvals[k] = (my - (double) (ptsy[k])) * sy + dy;
        success = bindings->cferbind->drawPoints(bindings->cferbind,
                                      xvals, yvals, numpts, symbolobj,
                                      colorobj, (double) ptsize);
        FerMem_Free(xvals, __FILE__, __LINE__);
        if ( ! success ) {
            /* grdelerrmsg is already assigned */
            return 0;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        xtuple = PyTuple_New( (Py_ssize_t) numpts );
        if ( xtuple == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawPoints: problems creating "
                                "a Python tuple");
            return 0;
        }
        for (k = 0; k < numpts; k++) {
            transval = (double) (ptsx[k]) * sx + dx;
            fltobj = PyFloat_FromDouble(transval);
            if ( fltobj == NULL ) {
                PyErr_Clear();
                strcpy(grdelerrmsg, "grdelDrawPoints: problems creating "
                                    "a Python float");
                Py_DECREF(xtuple);
                return 0;
            }
            /* PyTuple_SET_ITEM steals the reference to fltobj */
            PyTuple_SET_ITEM(xtuple, (Py_ssize_t) k, fltobj);
        }

        ytuple = PyTuple_New( (Py_ssize_t) numpts );
        if ( ytuple == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawPoints: problems creating "
                                "a Python tuple");
            Py_DECREF(xtuple);
            return 0;
        }
        for (k = 0; k < numpts; k++) {
            transval = (my - (double) (ptsy[k])) * sy + dy;
            fltobj = PyFloat_FromDouble(transval);
            if ( fltobj == NULL ) {
                PyErr_Clear();
                strcpy(grdelerrmsg, "grdelDrawPoints: problems creating "
                                    "a Python float");
                Py_DECREF(ytuple);
                Py_DECREF(xtuple);
                return 0;
            }
            /* PyTuple_SET_ITEM steals the reference to fltobj */
            PyTuple_SET_ITEM(ytuple, (Py_ssize_t) k, fltobj);
        }

        /*
         * Call the drawPoints method of the bindings instance.
         * Using 'N' to steal the reference to xtuple and to ytuple.
         */
        result = PyObject_CallMethod(bindings->pyobject, "drawPoints",
                          "NNOOd", xtuple, ytuple, (PyObject *) symbolobj,
                          (PyObject *) colorobj, (double) ptsize);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelDrawPoints: error when calling the Python "
                    "binding's drawPoints method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdeldrawPoints: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Draws a polygon.
 *
 * Arguments:
 *     window: Window with an active View to draw in
 *     ptsx: user X-coordinates of the vertices
 *     ptsy: user Y-coordinates of the vertices
 *     numpts: number of coordinates in ptsx and ptsy to use
 *     brush: Brush to use to fill the polygon; if NULL,
 *             the polygon will not be filled
 *     pen: Pen to use to outline the polygon; if NULL
 *             the polygon will not be outlined
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawPolygon(grdelType window, const float ptsx[],
               const float ptsy[], int numpts, grdelType brush,
               grdelType pen)
{
    const BindObj *bindings;
    grdelType brushobj;
    grdelType penobj;
    double   *xvals;
    double   *yvals;
    grdelBool success;
    PyObject *xtuple;
    PyObject *ytuple;
    PyObject *fltobj;
    PyObject *result;
    double my, sx, sy, dx, dy;
    double transval;
    int k;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelDrawPolygon called: "
            "window = %p, brush = %p, pen = %p, numpts = %d\n", 
            window, brush, pen, numpts);
    for (k = 0; k < numpts; k++)
        fprintf(debuglogfile, "   pt[%d] = (%f,%f)\n", k, ptsx[k], ptsy[k]);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawPolygon: window argument is not "
                            "a grdel Window");
        return 0;
    }
    if ( (brush == NULL) && (pen == NULL) ) {
        strcpy(grdelerrmsg, "grdelDrawPolygon: neither a pen nor "
                            "a brush was specified");
        return 0;
    }
    if ( brush != NULL ) {
        brushobj = grdelBrushVerify(brush, window);
        if ( brushobj == NULL ) {
            strcpy(grdelerrmsg, "grdelDrawPolygon: brush argument is not "
                                "a valid grdel Brush for the window");
            return 0;
        }
    }
    else
        brushobj = NULL;
    if ( pen != NULL ) {
        penobj = grdelPenVerify(pen, window);
        if ( penobj == NULL ) {
            strcpy(grdelerrmsg, "grdelDrawPolygon: pen argument is not "
                                "a valid grdel Pen for the window");
            return 0;
        }
    }
    else
        penobj = NULL;
    if ( numpts <= 2 ) {
        strcpy(grdelerrmsg, "grdelDrawPolygon: invalid number of points");
        return (grdelBool) 0;
    }

    /* Get the transform values for converting user to device coordinates */
    grdelGetTransformValues(&my, &sx, &sy, &dx, &dy);

    if ( bindings->cferbind != NULL ) {
        xvals = (double *) FerMem_Malloc(2 * numpts * sizeof(double), __FILE__, __LINE__);
        if ( xvals == NULL ) {
            sprintf(grdelerrmsg, "grdelDrawPolygon: out of memory "
                                 "for an array of %d doubles", 2 * numpts);
            return 0;
        }
        yvals = &(xvals[numpts]);
        for (k = 0; k < numpts; k++)
            xvals[k] = (double) (ptsx[k]) * sx + dx;
        for (k = 0; k < numpts; k++)
            yvals[k] = (my - (double) (ptsy[k])) * sy + dy;
        success = bindings->cferbind->drawPolygon(bindings->cferbind,
                                      xvals, yvals, numpts, brushobj, penobj);
        FerMem_Free(xvals, __FILE__, __LINE__);
        if ( ! success ) {
            /* grdelerrmsg is already assigned */
            return 0;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        xtuple = PyTuple_New( (Py_ssize_t) numpts );
        if ( xtuple == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawPolygon: problems creating "
                                "a Python tuple");
            return 0;
        }
        for (k = 0; k < numpts; k++) {
            transval = (double) (ptsx[k]) * sx + dx;
            fltobj = PyFloat_FromDouble(transval);
            if ( fltobj == NULL ) {
                PyErr_Clear();
                strcpy(grdelerrmsg, "grdelDrawPolygon: problems creating "
                                    "a Python float");
                Py_DECREF(xtuple);
                return 0;
            }
            /* PyTuple_SET_ITEM steals the reference to fltobj */
            PyTuple_SET_ITEM(xtuple, (Py_ssize_t) k, fltobj);
        }

        ytuple = PyTuple_New( (Py_ssize_t) numpts );
        if ( ytuple == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawPolygon: problems creating "
                                "a Python tuple");
            Py_DECREF(xtuple);
            return 0;
        }
        for (k = 0; k < numpts; k++) {
            transval = (my - (double) (ptsy[k])) * sy + dy;
            fltobj = PyFloat_FromDouble(transval);
            if ( fltobj == NULL ) {
                PyErr_Clear();
                strcpy(grdelerrmsg, "grdelDrawPolygon: problems creating "
                                    "a Python float");
                Py_DECREF(ytuple);
                Py_DECREF(xtuple);
                return 0;
            }
            /* PyTuple_SET_ITEM steals the reference to fltobj */
            PyTuple_SET_ITEM(ytuple, (Py_ssize_t) k, fltobj);
        }

        if ( brushobj == NULL )
            brushobj = Py_None;
        if ( penobj == NULL )
            penobj = Py_None;
        /*
         * Call the drawPolygon method of the bindings instance.
         * Using 'N' to steal the reference to xtuple and to ytuple.
         */
        result = PyObject_CallMethod(bindings->pyobject, "drawPolygon",
                          "NNOO", xtuple, ytuple, 
                          (PyObject *) brushobj, (PyObject *) penobj);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelDrawPolygon: error when calling the Python "
                    "binding's drawPolygon method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdeldrawPolygon: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Draws a rectangle.
 *
 * Arguments:
 *     window: Window with an active View to draw in
 *     left: user X-coordinate of the left edge
 *     bottom: user Y-coordinate of the bottom edge
 *     right: user X-coordinate of the right edge
 *     top: user Y-coordinate of the top edge
 *     brush: Brush to use to fill the rectangle; if NULL,
 *             the rectangle will not be filled
 *     pen: Pen to use to outline the rectangle; if NULL
 *             the rectangle will not be outlined
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawRectangle(grdelType window, float left, float bottom,
               float right, float top, grdelType brush, grdelType pen)
{
    const BindObj *bindings;
    grdelType brushobj;
    grdelType penobj;
    grdelBool success;
    PyObject *result;
    double my, sx, sy, dx, dy;
    double trlft, trbtm, trrgt, trtop;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelDrawRectangle called: "
            "window = %p, brush = %p, pen = %p\n"
            "   left = %f, bottom = %f, right = %f, top = %f\n", 
            window, brush, pen, left, bottom, right, top);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawRectangle: window argument is not "
                            "a grdel Window");
        return 0;
    }
    if ( (brush == NULL) && (pen == NULL) ) {
        strcpy(grdelerrmsg, "grdelDrawRectangle: neither a pen nor "
                            "a brush was specified");
        return (grdelBool) 0;
    }
    if ( brush != NULL ) {
        brushobj = grdelBrushVerify(brush, window);
        if ( brushobj == NULL ) {
            strcpy(grdelerrmsg, "grdelDrawRectangle: brush argument is not "
                                "a valid grdel Brush for the window");
            return (grdelBool) 0;
        }
    }
    else
        brushobj = NULL;
    if ( pen != NULL ) {
        penobj = grdelPenVerify(pen, window);
        if ( penobj == NULL ) {
            strcpy(grdelerrmsg, "grdelDrawRectangle: pen argument is not "
                                "a valid grdel Pen for the window");
            return (grdelBool) 0;
        }
    }
    else
        penobj = NULL;

    /* Get the transform values for converting user to device coordinates */
    grdelGetTransformValues(&my, &sx, &sy, &dx, &dy);
    trlft = (double) left * sx + dx;
    trrgt = (double) right * sx + dx;
    trtop = (my - (double) top) * sy + dy;
    trbtm = (my - (double) bottom) * sy + dy;

    if ( bindings->cferbind != NULL ) {
        success = bindings->cferbind->drawRectangle(bindings->cferbind,
                                      trlft, trbtm, trrgt, trtop,
                                      brushobj, penobj);
        if ( ! success ) {
            /* grdelerrmsg is already assigned */
            return 0;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        if ( brushobj == NULL )
            brushobj = Py_None;
        if ( penobj == NULL )
            penobj = Py_None;
        /*
         * Call the drawRectangle method of the bindings instance.
         */
        result = PyObject_CallMethod(bindings->pyobject, "drawRectangle",
                          "ddddOO", trlft, trbtm, trrgt, trtop,
                          (PyObject *) brushobj, (PyObject *) penobj);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelDrawRectangle: error when calling the Python "
                    "binding's drawRectangle method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdeldrawRectangle: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}


/*
 * Returns the size of given text if drawn with a given font.
 * The width is such that continuing text should be positioned 
 * at the start of this text plus this width.  The height will 
 * always be the ascent plus descent for the font and is 
 * independent of the text.
 *
 * Input Arguments:
 *     window: Window to use
 *     text: text string to use
 *     textlen: actual length of the text string
 *     font: font to use
 * Output Arguments:
 *     fltwidthptr: assigned the width of the text, in user 
 *               coordinates, if drawn in the given font
 *     fltheightptr: assigned the height of the text, in user 
 *                coordinates, if drawn in the given font
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelTextSize(grdelType window, const char *text, int textlen, 
                        grdelType font, float *fltwidthptr, float *fltheightptr)
{
    const BindObj *bindings;
    grdelType fontobj;
    grdelBool success;
    double width, height;
    PyObject *result;
    double my, sx, sy, dx, dy;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelTextSize called: "
            "window = %p, font = %p, text = '%.*s'\n", window, font, textlen, text);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelTextSize: window argument is not "
                            "a grdel Window");
        return 0;
    }
    fontobj = grdelFontVerify(font, window);
    if ( fontobj == NULL ) {
        strcpy(grdelerrmsg, "grdelTextSize: font argument is not "
                            "a valid grdel Font for the window");
        return 0;
    }

    if ( bindings->cferbind != NULL ) {
         success = bindings->cferbind->textSize(bindings->cferbind,
                             text, textlen, fontobj, &width, &height);
        if ( success == 0 ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        /* Call the textSize method of the bindings instance. */
        result = PyObject_CallMethod(bindings->pyobject, "textSize",
                          "s#O", text, textlen, (PyObject *) fontobj);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelTextSize: Error when calling the Python "
                    "binding's textSize method: %s", pyefcn_get_error());
            return 0;
        }
        if ( ! PyArg_ParseTuple(result, "dd", &width, &height) ) {
            Py_DECREF(result);
            sprintf(grdelerrmsg, "grdelTextSize: Error when parsing the Python "
                                 "binding's textSize return value: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelTextSize: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    /* Get the transform values for converting user to device coordinates */
    grdelGetTransformValues(&my, &sx, &sy, &dx, &dy);

    /* Convert the width and height back to user coordinates - just scaling, no offset */
    *fltwidthptr = (float) (width / sx);
    *fltheightptr = (float) (height / sy);

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelTextSize reponse: width = %f, height = %f\n", 
                          *fltwidthptr, *fltheightptr);
    fflush(debuglogfile);
#endif

    return 1;
}


/*
 * Draws text.
 *
 * Arguments:
 *     window: Window with an active View to draw in
 *     text: text string to draw
 *     textlen: actual length of the text string
 *     startx: user X-coordinate of the beginning of the text baseline 
 *     starty: user Y-coordinate of the beginning of the text baseline
 *     font: font to use
 *     color: color to use (as a solid brush or pen)
 *     rotate: angle of the text baseline in degrees
 *              clockwise from horizontal
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawText(grdelType window, const char *text, int textlen,
               float startx, float starty, grdelType font, grdelType color,
               float rotate)
{
    const BindObj *bindings;
    grdelType fontobj;
    grdelType colorobj;
    grdelBool success;
    PyObject *result;
    double my, sx, sy, dx, dy;
    double trstx, trsty;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelDrawText called: "
            "window = %p, font = %p, color = %p, rotate = %f, text = '%.*s'\n", 
             window, font, color, rotate, textlen, text);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawText: window argument is not "
                            "a grdel Window");
        return 0;
    }
    fontobj = grdelFontVerify(font, window);
    if ( fontobj == NULL ) {
        strcpy(grdelerrmsg, "grdelDrawText: font argument is not "
                            "a valid grdel Font for the window");
        return 0;
    }
    colorobj = grdelColorVerify(color, window);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "grdelDrawText: color argument is not "
                            "a valid grdel Color for the window");
        return 0;
    }

    /* Get the transform values for converting user to device coordinates */
    grdelGetTransformValues(&my, &sx, &sy, &dx, &dy);
    trstx = (double) startx * sx + dx;
    trsty = (my - (double) starty) * sy + dy;

    if ( bindings->cferbind != NULL ) {
        success = bindings->cferbind->drawText(bindings->cferbind,
                                      text, textlen, trstx, trsty,
                                      fontobj, colorobj, (double) rotate);
        if ( ! success ) {
            /* grdelerrmsg is already assigned */
            return 0;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        /* Call the drawText method of the bindings instance. */
        result = PyObject_CallMethod(bindings->pyobject, "drawText",
                          "s#ddOOd", text, textlen, trstx, trsty,
                          (PyObject *) fontobj, (PyObject *) colorobj,
                          (double) rotate);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelDrawText: Error when calling the Python "
                    "binding's drawText method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelDrawText: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Draws connected line segments.
 *
 * Input Arguments:
 *     window: Window with an active View to draw in
 *     ptsx: user X-coordinates of the endpoints
 *     ptsy: user Y-coordinates of the endpoints
 *     numpts: number of coordinates in ptsx and ptsy to use
 *     pen: Pen to use to draw the line segments
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgddrawmultiline_(int *success, void **window, float ptsx[],
               float ptsy[], int *numpts, void **pen)
{
    grdelBool result;

    result = grdelDrawMultiline(*window, ptsx, ptsy, *numpts, *pen);
    *success = result;
}

/*
 * Draws discrete points.
 *
 * Input Arguments:
 *     window: Window with an active View to draw in
 *     ptsx: user X-coordinates of the points
 *     ptsy: user Y-coordinates of the points
 *     numpts: number of coordinates in ptsx and ptsy to use
 *     symbol: Symbol to use to draw a point
 *     color: color of the Symbol
 *     ptsize: size of the symbol (scales with view size)
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgddrawpoints_(int *success, void **window, float ptsx[],
                    float ptsy[], int *numpts, void **symbol,
                    void **color, float *ptsize)
{
    grdelBool result;

    result = grdelDrawPoints(*window, ptsx, ptsy, *numpts, *symbol,
                             *color, *ptsize);
    *success = result;
}

/*
 * Draws a polygon.
 *
 * Input Arguments:
 *     window: Window with an active View to draw in
 *     ptsx: user X-coordinates of the vertices
 *     ptsy: user Y-coordinates of the vertices 
 *     numpts: number of coordinates in ptsx and ptsy to use
 *     brush: Brush to use to fill the polygon; if NULL,
 *             the polygon will not be filled
 *     pen: Pen to use to outline the polygon; if NULL
 *             the polygon will not be outlined
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgddrawpolygon_(int *success, void **window, float ptsx[],
                     float ptsy[], int *numpts, void **brush, void **pen)
{
    grdelBool result;

    result = grdelDrawPolygon(*window, ptsx, ptsy, *numpts, *brush, *pen);
    *success = result;
}

/*
 * Draws a rectangle.
 *
 * Input Arguments:
 *     window: Window with an active View to draw in
 *     left: user X-coordinate of the left edge 
 *     bottom: user Y-coordinate of the bottom edge 
 *     right: user X-coordinate of the right edge 
 *     top: user Y-coordinate of the top edge 
 *     brush: Brush to use to fill the rectangle; if NULL,
 *             the rectangle will not be filled
 *     pen: Pen to use to outline the rectangle; if NULL
 *             the rectangle will not be outlined
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgddrawrect_(int *success, void **window, float *left, float *bottom,
                  float *right, float *top, void **brush, void **pen)
{
    grdelBool result;

    result = grdelDrawRectangle(*window, *left, *bottom, *right, *top,
                                *brush, *pen);
    *success = result;
}

/*
 * Returns the size of given text if drawn with a given font.
 * The width is such that continuing text should be positioned 
 * at the text start plus this width.  The height will always 
 * be the ascent plus descent for the font and is independent 
 * of the text.
 *
 * Input Arguments:
 *     window: Window to use
 *     text: text string to use
 *     textlen: actual length of the text string
 *     font: font to use
 *
 * Output Arguments:
 *     width: width of the text, in user coordinates, 
 *            if drawn in the given font
 *     height: height of the text, in user coordinates, 
 *             if drawn in the given font
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdtextsize_(int *success, void **window, char *text, int *textlen,
                     void **font, float *width, float *height)
{
    grdelBool result;

    result = grdelTextSize(*window, text, *textlen, *font, width, height);
    *success = result;
}

/*
 * Draws text.
 *
 * Input Arguments:
 *     window: the Window with an active View to draw in
 *     text: text string to draw
 *     textlen: actual length of the text string
 *     startx: user X-coordinate of the beginning baseline
 *              of the text 
 *     starty: user Y-coordinate of the beginning baseline
 *              of the text 
 *     font: font to use 
 *     color: color to use (as a solid brush or pen)
 *     rotate: angle of the baseline in degrees
 *              clockwise from horizontal
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgddrawtext_(int *success, void **window, char *text, int *textlen,
                  float *startx, float *starty, void **font, void **color,
                  float *rotate)
{
    grdelBool result;

    result = grdelDrawText(*window, text, *textlen, *startx, *starty,
                           *font, *color, *rotate);
    *success = result;
}

