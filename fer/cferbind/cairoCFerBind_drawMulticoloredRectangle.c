/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Draw a rectangle filled using an array of colors.
 *
 * Arguments:
 *     left    - left edge coordinate of the rectangle
 *     bottom  - bottom edge coordinate of the rectangle
 *     right   - right edge coordinate of the rectangle
 *     top     - top edge coordinate of the rectangle
 *     numrows - number of rows to subdivide the rectangle
 *     numcols - number of columns to subdivide the rectangle
 *     colors  - a column-major listing of the colors to
 *               use (as solid colors) to fill the rectangle
 *               subdivisions.
 *
 * Coordinates are in units of pixels from te upper left
 * corner, increasing as one goes to the bottom right corner.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_drawMulticoloredRectangle(CFerBind *self,
                        double left, double bottom, double right,
                        double top, int numrows, int numcols,
                        grdelType colors[])
{
    CairoCFerBindData *instdata;
    CCFBColor *colorobj;
    int        j, k, q;
    double     unitfactor;
    double     adjleft;
    double     adjtop;
    double     adjwidth;
    double     adjheight;

    /* Sanity checks */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_drawMulticoloredRectangle: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    if ( instdata->context == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawMulticoloredRectangle: unexpected error, "
                            "NULL context");
        return 0;
    }
    for (q = 0; q < numrows * numcols; q++) {
        colorobj = (CCFBColor *) colors[q];
        if ( colorobj->id != CCFBColorId ) {
            sprintf(grdelerrmsg, "cairoCFerBind_drawMulticoloredRectangle: unexpected error, "
                                 "colors[%d] is not CCFBColor struct", q);
            return 0;
        }
    }

    /* Convertions factor for those surfaces that expect points instead of pixels */
    switch( instdata->imageformat ) {
    case CCFBIF_PDF:
    case CCFBIF_EPS:
    case CCFBIF_SVG:
        unitfactor = CCFB_POINTS_PER_PIXEL;
        break;
    default:
        unitfactor = 1.0;
        break;
    }

    /*
     * adjleft and adjtop are for the full rectangle,
     * adjwidth and adjheight are for the rectangles subdivisions
     */
    adjleft = left * unitfactor;
    adjtop = top * unitfactor;
    adjwidth = (right - left) * unitfactor / (double) numcols;
    adjheight = (bottom - top) * unitfactor / (double) numrows;

    /* initialize the context */
    cairo_set_line_width(instdata->context, unitfactor);
    cairo_set_dash(instdata->context, NULL, 0, 0.0);
    cairo_set_line_cap(instdata->context, CAIRO_LINE_CAP_SQUARE);
    cairo_set_line_join(instdata->context, CAIRO_LINE_JOIN_BEVEL);
    cairo_new_path(instdata->context);

    for (j = 0, q = 0; j < numcols; j++) {
        for (k = 0; k < numrows; k++, q++) {
            /* Create the path for this rectangle subdivision */
            cairo_rectangle(instdata->context, adjleft + j * adjwidth,
                            adjtop + k * adjheight, adjwidth, adjheight);
            /* Assign the solid color to use */
            colorobj = (CCFBColor *) colors[q];
            cairo_set_source_rgba(instdata->context, colorobj->redfrac,
                  colorobj->greenfrac, colorobj->bluefrac, colorobj->opaquefrac);
            /* Fill the rectangle, but preserve the path for stroking */
            cairo_fill_preserve(instdata->context);
            /* stroke and remove the path */
            cairo_stroke(instdata->context);
        }
    }

    return 1;
}

