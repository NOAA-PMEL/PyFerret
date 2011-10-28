/*
 * Drawing commands
 */
#include <Python.h> /* make sure Python.h is first */
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "pyferret.h"

/*
 * Draws connected line segments.
 *
 * Arguments:
 *     window: the Window with an active View to draw in
 *     ptsx: the X-coordinates of the points in View units
 *     ptsy: the Y-coordinates of the points in View units
 *     numpts: the number of coordinates in ptsx and ptsy to use
 *     pen: the Pen to use to draw the line segments
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawMultiline(grdelType window, const float ptsx[],
               const float ptsy[], int numpts, grdelType pen)
{
    PyObject *bindings;
    PyObject *penobj;
    PyObject *xtuple;
    PyObject *ytuple;
    PyObject *fltobj;
    PyObject *result;
    int k;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelDrawMultiline called: "
            "window = %X, pen = %X\n", window, pen);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawMultiline: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    penobj = grdelPenVerify(pen, window);
    if ( penobj == NULL ) {
        strcpy(grdelerrmsg, "grdelDrawMultiline: pen argument is not "
                            "a valid grdel Pen for the window");
        return (grdelBool) 0;
    }
    if ( numpts <= 1 ) {
        strcpy(grdelerrmsg, "grdelDrawMultiline: invalid number of points");
        return (grdelBool) 0;
    }

    xtuple = PyTuple_New( (Py_ssize_t) numpts );
    if ( xtuple == NULL ) {
        PyErr_Clear();
        strcpy(grdelerrmsg, "grdelDrawMultiline: problems creating "
                            "a Python tuple");
        return (grdelBool) 0;
    }
    for (k = 0; k < numpts; k++) {
        fltobj = PyFloat_FromDouble( (double) (ptsx[k]) );
        if ( fltobj == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawMultiline: problems creating "
                                "a Python float");
            Py_DECREF(xtuple);
            return (grdelBool) 0;
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
        return (grdelBool) 0;
    }
    for (k = 0; k < numpts; k++) {
        fltobj = PyFloat_FromDouble( (double) (ptsy[k]) );
        if ( fltobj == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawMultiline: problems creating "
                                "a Python float");
            Py_DECREF(ytuple);
            Py_DECREF(xtuple);
            return (grdelBool) 0;
        }
        /* PyTuple_SET_ITEM steals the reference to fltobj */
        PyTuple_SET_ITEM(ytuple, (Py_ssize_t) k, fltobj);
    }

    /*
     * Call the drawMultiline method of the bindings instance.
     * Using 'N' to steal the reference to xtuple and to ytuple.
     */
    result = PyObject_CallMethod(bindings, "drawMultiline", "NNO",
                                 xtuple, ytuple, penobj);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelDrawMultiline: error when calling "
                "the binding's drawMultiline method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Draws discrete points.
 *
 * Arguments:
 *     window: the Window with an active View to draw in
 *     ptsx: the X-coordinates of the points in View units
 *     ptsy: the Y-coordinates of the points in View units
 *     numpts: the number of coordinates in ptsx and ptsy to use
 *     symbol: the Symbol to use to draw a point
 *     color: color of the Symbol
 *     ptsize: size of the symbol in View units
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawPoints(grdelType window, const float ptsx[],
               const float ptsy[], int numpts, grdelType symbol,
               grdelType color, float ptsize)
{
    PyObject *bindings;
    PyObject *symbolobj;
    PyObject *colorobj;
    PyObject *xtuple;
    PyObject *ytuple;
    PyObject *fltobj;
    PyObject *result;
    int k;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelDrawPoints called: "
            "window = %X, symbol = %X, color = %X", window, symbol, color);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawPoints: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    symbolobj = grdelSymbolVerify(symbol, window);
    if ( symbolobj == NULL ) {
        strcpy(grdelerrmsg, "grdelDrawPoints: symbol argument is not "
                            "a valid grdel Symbol for the window");
        return (grdelBool) 0;
    }
    colorobj = grdelColorVerify(color, window);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "grdelDrawPoints: color argument is not "
                            "a valid grdel Color for the window");
        return (grdelBool) 0;
    }
    if ( numpts <= 0 ) {
        strcpy(grdelerrmsg, "grdelDrawPoints: invalid number of points");
        return (grdelBool) 0;
    }

    xtuple = PyTuple_New( (Py_ssize_t) numpts );
    if ( xtuple == NULL ) {
        PyErr_Clear();
        strcpy(grdelerrmsg, "grdelDrawPoints: problems creating "
                            "a Python tuple");
        return (grdelBool) 0;
    }
    for (k = 0; k < numpts; k++) {
        fltobj = PyFloat_FromDouble( (double) (ptsx[k]) );
        if ( fltobj == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawPoints: problems creating "
                                "a Python float");
            Py_DECREF(xtuple);
            return (grdelBool) 0;
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
        return (grdelBool) 0;
    }
    for (k = 0; k < numpts; k++) {
        fltobj = PyFloat_FromDouble( (double) (ptsy[k]) );
        if ( fltobj == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawPoints: problems creating "
                                "a Python float");
            Py_DECREF(ytuple);
            Py_DECREF(xtuple);
            return (grdelBool) 0;
        }
        /* PyTuple_SET_ITEM steals the reference to fltobj */
        PyTuple_SET_ITEM(ytuple, (Py_ssize_t) k, fltobj);
    }

    /*
     * Call the drawPoints method of the bindings instance.
     * Using 'N' to steal the reference to xtuple and to ytuple.
     */
    result = PyObject_CallMethod(bindings, "drawPoints", "NNOOd",
                                 xtuple, ytuple, symbolobj,
                                 colorobj, (double) ptsize);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelDrawPoints: error when calling "
                "the binding's drawPoints method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Draws a polygon.
 *
 * Arguments:
 *     window: the Window with an active View to draw in
 *     ptsx: the X-coordinates of the points in View units
 *     ptsy: the Y-coordinates of the points in View units
 *     numpts: the number of coordinates in ptsx and ptsy to use
 *     brush: the Brush to use to fill the polygon; if NULL,
 *             the polygon will not be filled
 *     pen: the Pen to use to outline the polygon; if NULL
 *             the polygon will not be outlined
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawPolygon(grdelType window, const float ptsx[],
               const float ptsy[], int numpts, grdelType brush,
               grdelType pen)
{
    PyObject *bindings;
    PyObject *brushobj;
    PyObject *penobj;
    PyObject *xtuple;
    PyObject *ytuple;
    PyObject *fltobj;
    PyObject *result;
    int k;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelDrawPolygon called: "
            "window = %X, brush = %X, pen = %X\n", window, brush, pen);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawPolygon: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    if ( (brush == NULL) && (pen == NULL) ) {
        strcpy(grdelerrmsg, "grdelDrawPolygon: neither a pen nor "
                            "a brush was specified");
        return (grdelBool) 0;
    }
    if ( brush != NULL ) {
        brushobj = grdelBrushVerify(brush, window);
        if ( brushobj == NULL ) {
            strcpy(grdelerrmsg, "grdelDrawPolygon: brush argument is not "
                                "a valid grdel Brush for the window");
            return (grdelBool) 0;
        }
    }
    else
        brushobj = Py_None;
    if ( pen != NULL ) {
        penobj = grdelPenVerify(pen, window);
        if ( penobj == NULL ) {
            strcpy(grdelerrmsg, "grdelDrawPolygon: pen argument is not "
                                "a valid grdel Pen for the window");
            return (grdelBool) 0;
        }
    }
    else
        penobj = Py_None;
    if ( numpts <= 2 ) {
        strcpy(grdelerrmsg, "grdelDrawPolygon: invalid number of points");
        return (grdelBool) 0;
    }

    xtuple = PyTuple_New( (Py_ssize_t) numpts );
    if ( xtuple == NULL ) {
        PyErr_Clear();
        strcpy(grdelerrmsg, "grdelDrawPolygon: problems creating "
                            "a Python tuple");
        return (grdelBool) 0;
    }
    for (k = 0; k < numpts; k++) {
        fltobj = PyFloat_FromDouble( (double) (ptsx[k]) );
        if ( fltobj == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawPolygon: problems creating "
                                "a Python float");
            Py_DECREF(xtuple);
            return (grdelBool) 0;
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
        return (grdelBool) 0;
    }
    for (k = 0; k < numpts; k++) {
        fltobj = PyFloat_FromDouble( (double) (ptsy[k]) );
        if ( fltobj == NULL ) {
            PyErr_Clear();
            strcpy(grdelerrmsg, "grdelDrawPolygon: problems creating "
                                "a Python float");
            Py_DECREF(ytuple);
            Py_DECREF(xtuple);
            return (grdelBool) 0;
        }
        /* PyTuple_SET_ITEM steals the reference to fltobj */
        PyTuple_SET_ITEM(ytuple, (Py_ssize_t) k, fltobj);
    }

    /*
     * Call the drawPolygon method of the bindings instance.
     * Using 'N' to steal the reference to xtuple and to ytuple.
     */
    result = PyObject_CallMethod(bindings, "drawPolygon", "NNOO",
                                 xtuple, ytuple, brushobj, penobj);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelDrawPolygon: error when calling "
                "the binding's drawPolygon method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Draws a rectangle.
 *
 * Arguments:
 *     window: the Window with an active View to draw in
 *     left: the X-coordinate of the left edge in View units
 *     bottom: the Y-coordinate of the bottom edge in View units
 *     right: the X-coordinate of the right edge in View units
 *     top: the Y-coordinate of the top edge in View units
 *     brush: the Brush to use to fill the rectangle; if NULL,
 *             the rectangle will not be filled
 *     pen: the Pen to use to outline the rectangle; if NULL
 *             the rectangle will not be outlined
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawRectangle(grdelType window, float left, float bottom,
               float right, float top, grdelType brush, grdelType pen)
{
    PyObject *bindings;
    PyObject *brushobj;
    PyObject *penobj;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelDrawRectangle called: "
            "window = %X, brush = %X, pen = %X\n", window, brush, pen);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawRectangle: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
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
        brushobj = Py_None;
    if ( pen != NULL ) {
        penobj = grdelPenVerify(pen, window);
        if ( penobj == NULL ) {
            strcpy(grdelerrmsg, "grdelDrawRectangle: pen argument is not "
                                "a valid grdel Pen for the window");
            return (grdelBool) 0;
        }
    }
    else
        penobj = Py_None;

    /*
     * Call the drawRectangle method of the bindings instance.
     */
    result = PyObject_CallMethod(bindings, "drawRectangle", "ddddOO",
                                 (double) left, (double) bottom,
                                 (double) right, (double) top,
                                 brushobj, penobj);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelDrawRectangle: error when calling "
                "the binding's drawRectangle method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Draws a filled rectangle using an array of solid colors.
 * The rectangle is divided into a given number of equally
 * spaced rows and a number of equally spaced columns.  Each
 * of these cells is then filled with a color (using a solid
 * brush) from the corresponding element in an array of colors.
 *
 * Arguments:
 *     window: the Window with an active View to draw in
 *     left: the X-coordinate of the left edge in View units
 *     bottom: the Y-coordinate of the bottom edge in View units
 *     right: the X-coordinate of the right edge in View units
 *     top: the Y-coordinate of the top edge in View units
 *     numrows: the number of equally spaced rows
 *              to subdivide the rectangle into
 *     numcols: the number of equally spaced columns
 *     	        to subdivide the rectangle into
 *     colors: a flattened column-major 2-D array of colors
 *              specifying the color of each corresponding cell
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawMulticolorRectangle(grdelType window,
               float left, float bottom, float right, float top,
               int numrows, int numcols, const grdelType colors[])
{
    PyObject *bindings;
    PyObject *colortuple;
    PyObject *colorobj;
    PyObject *result;
    int numcolors;
    int k;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelDrawMulticolorRectangle called: "
            "window = %X\n", window);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawMulticolor: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    numcolors = numrows * numcols;
    if ( (numrows <= 0)  || (numcols <= 0) || (numcolors <= 0) ) {
        strcpy(grdelerrmsg, "grdelDrawMulticolor: invalid numrows and/or "
                            "numcols value");
        return (grdelBool) 0;
    }
    colortuple = PyTuple_New( (Py_ssize_t) numcolors );
    if ( colortuple == NULL ) {
        PyErr_Clear();
        strcpy(grdelerrmsg, "grdelDrawMulticolor: problems creating "
                            "a Python tuple");
        return (grdelBool) 0;
    }
    for (k = 0; k < numrows * numcols; k++) {
        colorobj = grdelColorVerify(colors[k], window);
        if ( colorobj == NULL  ) {
            sprintf(grdelerrmsg, "grdelDrawMulticolor: colors[%d] is not "
                                 "a valid grdel Color for the window", k);
            Py_DECREF(colortuple);
            return (grdelBool) 0;
        }
        /*
         * PyTuple_SET_ITEM steals a reference to colorobj,
         * so increment the reference count on it.
         */
        Py_INCREF(colorobj);
        PyTuple_SET_ITEM(colortuple, (Py_ssize_t) k, colorobj);
    }

    /*
     * Call the drawMulticolorRectangle method of the bindings instance.
     * Using 'N' to steal the reference to colortuple.
     */
    result = PyObject_CallMethod(bindings, "drawMulticolorRectangle", "ddddiiN",
                                 (double) left, (double) bottom, 
                                 (double) right, (double) top, 
                                 numrows, numcols, colortuple);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelDrawMulticolor: error when calling "
                "the binding's drawMulticolorRectangle method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Draws text.
 *
 * Arguments:
 *     window: the Window with an active View to draw in
 *     text: the text string to draw
 *     textlen: actual length of the text string
 *     startx: the X-coordinate of the beginning baseline
 *              of the text in View units
 *     starty: the Y-coordinate of the beginning baseline
 *              of the text in View units
 *     font: the font to use for the text
 *     color: the color to use (as a solid brush or pen)
 *              for the text
 *     rotate: the angle of the baseline in degrees
 *              clockwise from horizontal
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelDrawText(grdelType window, const char *text, int textlen,
               float startx, float starty, grdelType font, grdelType color,
               float rotate)
{
    PyObject *bindings;
    PyObject *fontobj;
    PyObject *colorobj;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelDrawText called: "
            "window = %X, font = %X, color = %X\n", window, font, color);
    fflush(debuglogfile);
#endif

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL  ) {
        strcpy(grdelerrmsg, "grdelDrawText: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    fontobj = grdelFontVerify(font, window);
    if ( fontobj == NULL ) {
        strcpy(grdelerrmsg, "grdelDrawText: font argument is not "
                            "a valid grdel Font for the window");
        return (grdelBool) 0;
    }
    colorobj = grdelColorVerify(color, window);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "grdelDrawText: color argument is not "
                            "a valid grdel Color for the window");
        return (grdelBool) 0;
    }

    /* Call the drawText method of the bindings instance. */
    result = PyObject_CallMethod(bindings, "drawText", "s#ddOOd",
                          text, textlen, (double) startx, (double) starty,
                          fontobj, colorobj, (double) rotate);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelDrawText: Error when calling "
                "the binding's drawText method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Draws connected line segments.
 *
 * Input Arguments:
 *     window: the Window with an active View to draw in
 *     ptsx: the X-coordinates of the points in View units
 *     ptsy: the Y-coordinates of the points in View units
 *     numpts: the number of coordinates in ptsx and ptsy to use
 *     pen: the Pen to use to draw the line segments
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
 *     window: the Window with an active View to draw in
 *     ptsx: the X-coordinates of the points in View units
 *     ptsy: the Y-coordinates of the points in View units
 *     numpts: the number of coordinates in ptsx and ptsy to use
 *     symbol: the Symbol to use to draw a point
 *     color: color of the Symbol
 *     ptsize: size of the symbol in View units
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
 *     window: the Window with an active View to draw in
 *     ptsx: the X-coordinates of the points in View units
 *     ptsy: the Y-coordinates of the points in View units
 *     numpts: the number of coordinates in ptsx and ptsy to use
 *     brush: the Brush to use to fill the polygon; if NULL,
 *             the polygon will not be filled
 *     pen: the Pen to use to outline the polygon; if NULL
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
 *     window: the Window with an active View to draw in
 *     left: the X-coordinate of the left edge in View units
 *     bottom: the Y-coordinate of the bottom edge in View units
 *     right: the X-coordinate of the right edge in View units
 *     top: the Y-coordinate of the top edge in View units
 *     brush: the Brush to use to fill the rectangle; if NULL,
 *             the rectangle will not be filled
 *     pen: the Pen to use to outline the rectangle; if NULL
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
 * Draws a filled rectangle using an array of solid colors.
 * The rectangle is divided into a given number of equally
 * spaced rows and a number of equally spaced columns.  Each
 * of these cells is then filled with a color (using a solid
 * brush) from the corresponding element in an array of colors.
 *
 * Input Arguments:
 *     window: the Window with an active View to draw in
 *     left: the X-coordinate of the left edge in View units
 *     bottom: the Y-coordinate of the bottom edge in View units
 *     right: the X-coordinate of the right edge in View units
 *     top: the Y-coordinate of the top edge in View units
 *     numrows: the number of equally spaced rows
 *              to subdivide the rectangle into
 *     numcols: the number of equally spaced columns
 *              to subdivide the rectangle into
 *     colors: a flattened column-major 2-D array of colors
 *              specifying the color of each corresponding cell
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgddrawmulticolor_(int *success, void **window, float *left,
                        float *bottom, float *right, float *top,
                        int *numrows, int *numcols, void *colors[])
{
    grdelBool result;

    result = grdelDrawMulticolorRectangle(*window, *left, *bottom, *right,
                                          *top, *numrows, *numcols, colors);
    *success = result;
}

/*
 * Draws text.
 *
 * Input Arguments:
 *     window: the Window with an active View to draw in
 *     text: the text string to draw
 *     textlen: actual length of the text string
 *     startx: the X-coordinate of the beginning baseline
 *              of the text in View units
 *     starty: the Y-coordinate of the beginning baseline
 *              of the text in View units
 *     font: the font to use for the text
 *     color: the color to use (as a solid brush or pen)
 *              for the text
 *     rotate: the angle of the baseline in degrees
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

