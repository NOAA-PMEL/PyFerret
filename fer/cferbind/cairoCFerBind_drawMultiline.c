/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Draw a series of connected line segments to this "Window".
 *
 * Arguments:
 *     ptsx   - x-coordinates of the line segment endpoints
 *     ptsy   - y-coordinates of the line segment endpoints
 *     numpts - number of endpoints given in ptsx and ptsy
 *     pen    - Pen object to use for drawing the lines
 *
 * Coordinates are in units of pixels from te upper left
 * corner, increasing as one goes to the bottom right corner.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_drawMultiline(CFerBind *self, double ptsx[],
                                      double ptsy[], int numpts, grdelType pen)
{
    CairoCFerBindData *instdata;
    CCFBPen *penobj;
    double   unitfactor;
    double   xval, yval;
    int      k;
    double   adjwidth;
    double   adjdashes[8];

    /* Sanity checks */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_drawMultiline: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    if ( instdata->context == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawMultiline: unexpected error, "
                            "NULL context");
        return 0;
    }
    penobj = (CCFBPen *) pen;
    if ( penobj->id != CCFBPenId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawMultiline: unexpected error, "
                            "pen is not CCFBPen struct");
        return 0;
    }
    if ( numpts < 2 ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawMultiline: "
                            "fewer than two points given");
        return 0;
    }

    /* Convertions factor for those surfaces that expect points instead of pixels */
    switch( instdata->imageformat ) {
    case CCFBIF_PDF:
    case CCFBIF_PS:
    case CCFBIF_SVG:
        unitfactor = CCFB_POINTS_PER_PIXEL;
        break;
    default:
        unitfactor = 1.0;
        break;
    }

    /* Create the path that will be stroked */
    cairo_new_path(instdata->context);
    xval = ptsx[0] * unitfactor;
    yval = ptsy[0] * unitfactor;
    cairo_move_to(instdata->context, xval, yval);
    for (k = 1; k < numpts; k++) {
        xval = ptsx[k] * unitfactor;
        yval = ptsy[k] * unitfactor;
        cairo_line_to(instdata->context, xval, yval);
    }

    /* Assign the line color */
    cairo_set_source_rgba(instdata->context, penobj->color.redfrac,
                          penobj->color.greenfrac, penobj->color.bluefrac,
                          penobj->color.opaquefrac);
    /* Assign the adjusted line width */
    adjwidth = penobj->width * instdata->viewfactor;
    if ( adjwidth < 1.0 )
        adjwidth = 1.0;
    adjwidth *= unitfactor;
    cairo_set_line_width(instdata->context, adjwidth);
    /* Assign the line style (solid/dash/dor/dashdot) using the adjusted width */
    for (k = 0; k < penobj->numdashes; k++)
        adjdashes[k] = penobj->dashes[k] * adjwidth;
    cairo_set_dash(instdata->context, adjdashes, penobj->numdashes, 0.0);
    if ( cairo_status(instdata->context) != CAIRO_STATUS_SUCCESS ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawMultiline: unexpected error, "
                            "problems setting pen dashes");
        return 0;
    }
    /* Assign the line cap and join styles */
    cairo_set_line_cap(instdata->context, penobj->captype);
    cairo_set_line_join(instdata->context, penobj->jointype);

    /* stroke and remove the path */
    cairo_stroke(instdata->context);

    return 1;
}

