/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Draw discrete points in this "Window".
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_drawPoints(CFerBind *self, double ptsx[], double ptsy[],
                                   int numpts, grdelType symbol, grdelType color,
                                   double symsize)
{
    CairoCFerBindData *instdata;
    CCFBColor *colorobj;
    char   symchar;
    int    k;
    double unitfactor;
    double scalefactor;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawPoints: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    if ( instdata->context == NULL ) {
        /* Create the Cairo Surface and Context if they do not exist */
        if ( ! cairoCFerBind_createSurface(self) ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    colorobj = (CCFBColor *) color;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawPoints: unexpected error, "
                            "color is not CCFBColor struct");
        return 0;
    }

    /* Assign the (solid) color to use for the symbols */
    if ( instdata->usealpha )
        cairo_set_source_rgba(instdata->context, colorobj->redfrac,
                              colorobj->greenfrac, colorobj->bluefrac,
                              colorobj->opaquefrac);
    else
        cairo_set_source_rgb(instdata->context, colorobj->redfrac,
                             colorobj->greenfrac, colorobj->bluefrac);

    /* Convertions factor for those surfaces that expect points instead of pixels */
    if ( instdata->imageformat == CCFBIF_PNG ) {
        unitfactor = 1.0;
    }
    else {
        unitfactor = CCFB_POINTS_PER_PIXEL;
    }

    /* Scaling factor to use for these symbols "drawn" as 100x100 pixel paths */
    scalefactor = symsize * instdata->widthfactor;
    if ( scalefactor < 5.0 )
        scalefactor = 5.0;
    scalefactor *= unitfactor / 100.0;

    /* Assign the line style for drawing the symbols */
    cairo_set_line_width(instdata->context, 15.0 * scalefactor);
    cairo_set_dash(instdata->context, NULL, 0, 0.0);
    cairo_set_line_cap(instdata->context, CAIRO_LINE_CAP_SQUARE);
    cairo_set_line_join(instdata->context, CAIRO_LINE_JOIN_BEVEL);

    /* A symbol object is just a character cast as a pointer */
    symchar = (char) symbol;
    switch( symchar ) {
    case '.':
        cairo_new_path(instdata->context);
        for (k = 0; k < numpts; k++) {
            cairo_new_sub_path(instdata->context);
            cairo_arc(instdata->context,
                      ptsx[k] * unitfactor,
                      ptsy[k] * unitfactor,
                      20.0 * scalefactor,
                      0.0, 2.0 * M_PI);
            cairo_close_path(instdata->context);
        }
        cairo_fill(instdata->context);
        break;
    case 'o':
        cairo_new_path(instdata->context);
        for (k = 0; k < numpts; k++) {
            cairo_new_sub_path(instdata->context);
            cairo_arc(instdata->context,
                      ptsx[k] * unitfactor,
                      ptsy[k] * unitfactor,
                      40.0 * scalefactor,
                      0.0, 2.0 * M_PI);
            cairo_close_path(instdata->context);
        }
        cairo_stroke(instdata->context);
        break;
    case '+':
        cairo_new_path(instdata->context);
        for (k = 0; k < numpts; k++) {
            cairo_move_to(instdata->context,
                          ptsx[k] * unitfactor,
                          ptsy[k] * unitfactor - 40.0 * scalefactor);
            cairo_line_to(instdata->context,
                          ptsx[k] * unitfactor,
                          ptsy[k] * unitfactor + 40.0 * scalefactor);
            cairo_move_to(instdata->context,
                          ptsx[k] * unitfactor - 40.0 * scalefactor,
                          ptsy[k] * unitfactor);
            cairo_line_to(instdata->context,
                          ptsx[k] * unitfactor + 40.0 * scalefactor,
                          ptsy[k] * unitfactor);
        }
        cairo_stroke(instdata->context);
        break;
    case 'x':
        cairo_new_path(instdata->context);
        for (k = 0; k < numpts; k++) {
            cairo_move_to(instdata->context,
                          ptsx[k] * unitfactor - 30.0 * scalefactor,
                          ptsy[k] * unitfactor - 30.0 * scalefactor);
            cairo_line_to(instdata->context,
                          ptsx[k] * unitfactor + 30.0 * scalefactor,
                          ptsy[k] * unitfactor + 30.0 * scalefactor);
            cairo_move_to(instdata->context,
                          ptsx[k] * unitfactor - 30.0 * scalefactor,
                          ptsy[k] * unitfactor + 30.0 * scalefactor);
            cairo_line_to(instdata->context,
                          ptsx[k] * unitfactor + 30.0 * scalefactor,
                          ptsy[k] * unitfactor - 30.0 * scalefactor);
        }
        cairo_stroke(instdata->context);
        break;
    case '*':
        cairo_new_path(instdata->context);
        for (k = 0; k < numpts; k++) {
            cairo_move_to(instdata->context,
                          ptsx[k] * unitfactor,
                          ptsy[k] * unitfactor - 40.0 * scalefactor);
            cairo_line_to(instdata->context,
                          ptsx[k] * unitfactor,
                          ptsy[k] * unitfactor + 40.0 * scalefactor);
            cairo_move_to(instdata->context,
                          ptsx[k] * unitfactor - 34.641 * scalefactor,
                          ptsy[k] * unitfactor - 20.0   * scalefactor);
            cairo_line_to(instdata->context,
                          ptsx[k] * unitfactor + 34.641 * scalefactor,
                          ptsy[k] * unitfactor + 20.0   * scalefactor);
            cairo_move_to(instdata->context,
                          ptsx[k] * unitfactor - 34.641 * scalefactor,
                          ptsy[k] * unitfactor + 20.0   * scalefactor);
            cairo_line_to(instdata->context,
                          ptsx[k] * unitfactor + 34.641 * scalefactor,
                          ptsy[k] * unitfactor - 20.0   * scalefactor);
        }
        cairo_stroke(instdata->context);
        break;
    case '^':
        cairo_new_path(instdata->context);
        for (k = 0; k < numpts; k++) {
            cairo_move_to(instdata->context,
                          ptsx[k] * unitfactor - 40.0 * scalefactor,
                          ptsy[k] * unitfactor + 30.0 * scalefactor);
            cairo_line_to(instdata->context,
                          ptsx[k] * unitfactor,
                          ptsy[k] * unitfactor - 39.282 * scalefactor);
            cairo_line_to(instdata->context,
                          ptsx[k] * unitfactor + 40.0 * scalefactor,
                          ptsy[k] * unitfactor + 30.0 * scalefactor);
            cairo_close_path(instdata->context);
        }
        cairo_stroke(instdata->context);
        break;
    case '#':
        cairo_new_path(instdata->context);
        for (k = 0; k < numpts; k++) {
            cairo_rectangle(instdata->context,
                            ptsx[k] * unitfactor - 35.0 * scalefactor,
                            ptsy[k] * unitfactor - 35.0 * scalefactor,
                            70.0 * scalefactor,
                            70.0 * scalefactor);
        }
        cairo_stroke(instdata->context);
        break;
    default:
        sprintf(grdelerrmsg, "cairoCFerBind_drawPoints: unexpected error, "
                             "unknown symbol '%c'", symchar);
        return 0;
    }

    instdata->somethingdrawn = 1;
    instdata->imagechanged = 1;
    return 1;
}

