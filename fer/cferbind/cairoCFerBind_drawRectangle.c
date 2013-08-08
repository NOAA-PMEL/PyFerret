/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Draw a rectangle to this "Window".
 *
 * Arguments:
 *     left   - left edge coordinate of the rectangle
 *     bottom - bottom edge coordinate of the rectangle
 *     right  - right edge coordinate of the rectangle
 *     top    - top edge coordinate of the rectangle
 *     brush  - Brush object to use for filling the rectangle
 *     pen    - Pen object to use for drawing the rectangle edges
 *
 * Coordinates are in units of pixels from te upper left
 * corner, increasing as one goes to the bottom right corner.
 *
 * If the brush argument is NULL, the rectangle will not be
 * filled.  If the pen argument is NULL, the rectangle edges
 * will be drawn using a solid cosmetic pen with the same
 * color/pattern as the brush.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_drawRectangle(CFerBind *self, double left, double bottom,
                        double right, double top, grdelType brush, grdelType pen)
{
    CairoCFerBindData *instdata;
    CCFBBrush *brushobj;
    CCFBPen   *penobj;
    double     unitfactor;
    double     adjleft;
    double     adjtop;
    double     adjwidth;
    double     adjheight;
    int        k;
    double     adjdashes[8];
    int        antialias;
    /* cairo_matrix_t current_transform; */

    /* Sanity checks */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawRectangle: unexpected error, "
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
    if ( brush != NULL ) {
        brushobj = (CCFBBrush *) brush;
        if ( brushobj->id != CCFBBrushId ) {
            strcpy(grdelerrmsg, "cairoCFerBind_drawRectangle: unexpected error, "
                                "brush is not CCFBBrush struct");
            return 0;
        }
    }
    else
        brushobj = NULL;
    if ( pen != NULL ) {
        penobj = (CCFBPen *) pen;
        if ( penobj->id != CCFBPenId ) {
            strcpy(grdelerrmsg, "cairoCFerBind_drawRectangle: unexpected error, "
                                "pen is not CCFBPen struct");
            return 0;
        }
    }
    else
        penobj = NULL;
    if ( (brushobj == NULL) && (penobj == NULL) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawRectangle: "
                            "both brush and pen are NULL");
        return 0;
    }

    /* Convertions factor for those surfaces that expect points instead of pixels */
    if ( instdata->imageformat == CCFBIF_PNG ) {
        unitfactor = 1.0;
    }
    else {
        unitfactor = CCFB_POINTS_PER_PIXEL;
    }

    /* Turn off antialiasing for this operation */
    antialias = instdata->antialias;
    cairoCFerBind_setAntialias(self, 0);

    /* Create the path that will be filled and/or stroked */
    cairo_new_path(instdata->context);
    adjleft = left * unitfactor;
    adjtop = top * unitfactor;
    adjwidth = (right - left) * unitfactor;
    adjheight = (bottom - top) * unitfactor;
    cairo_rectangle(instdata->context, adjleft, adjtop, adjwidth, adjheight);

    if ( (brushobj != NULL) && (penobj == NULL) ) {
        /* Simultaneously add a cosmetic pen around the fill */
        /*
         * cairo_push_group(instdata->context); 
         */

        /* Clear any transformation so it is not applied twice */
        /*
         * cairo_get_matrix(instdata->context, &current_transform);
         * cairo_identity_matrix(instdata->context);
         */

        /* Draw with opaque colors in this group */

        if ( brushobj->pattern != NULL )
            cairo_set_source(instdata->context, brushobj->pattern);
        else if ( instdata->usealpha )
            cairo_set_source_rgba(instdata->context, brushobj->color.redfrac,
                  brushobj->color.greenfrac, brushobj->color.bluefrac,
                  brushobj->color.opaquefrac);
        else
            cairo_set_source_rgb(instdata->context, brushobj->color.redfrac,
                  brushobj->color.greenfrac, brushobj->color.bluefrac);

        /* Fill the rectangle, but preserve the path for stroking */
        cairo_fill_preserve(instdata->context);

        /* Stroke the path with a solid cosmetic line */
        cairo_set_line_width(instdata->context, unitfactor);
        cairo_set_dash(instdata->context, NULL, 0, 0.0);
        cairo_set_line_cap(instdata->context, CAIRO_LINE_CAP_SQUARE);
        cairo_set_line_join(instdata->context, CAIRO_LINE_JOIN_BEVEL);
        cairo_stroke(instdata->context);

        /* Reset the original transformation */
        /*
         * cairo_set_matrix(instdata->context, &current_transform);
         */

        /* Draw this group using the brush alpha value (if appropriate) */
        /*
         * cairo_pop_group_to_source(instdata->context);
         * if ( instdata->usealpha )
         *     cairo_paint_with_alpha(instdata->context, brushobj->color.opaquefrac);
         * else
         *     cairo_paint(instdata->context);
         */
    }
    else {
        /* First fill if requested */
        if ( brushobj != NULL ) {
            /* Fill pattern or solid color */
            if ( brushobj->pattern != NULL )
                cairo_set_source(instdata->context, brushobj->pattern);
            else if ( instdata->usealpha )
                cairo_set_source_rgba(instdata->context, brushobj->color.redfrac,
                      brushobj->color.greenfrac, brushobj->color.bluefrac,
                      brushobj->color.opaquefrac);
            else
                cairo_set_source_rgb(instdata->context, brushobj->color.redfrac,
                      brushobj->color.greenfrac, brushobj->color.bluefrac);
            /* Fill the rectangle, but preserve the path for stroking */
            cairo_fill_preserve(instdata->context);
        }

        /* Assign the line color to the context */
        if ( instdata->usealpha )
            cairo_set_source_rgba(instdata->context, penobj->color.redfrac,
                  penobj->color.greenfrac, penobj->color.bluefrac,
                  penobj->color.opaquefrac);
        else
            cairo_set_source_rgb(instdata->context, penobj->color.redfrac,
                  penobj->color.greenfrac, penobj->color.bluefrac);
        /* Assign the adjusted line width */
        adjwidth = penobj->width * instdata->widthfactor;
        /* width of zero is a cosmetic pen - make it 1 pixel wide */
        if ( adjwidth == 0.0 )
            adjwidth = 1.0;
        adjwidth *= unitfactor;
        cairo_set_line_width(instdata->context, adjwidth);
        /* Assign the line style (solid/dash/dor/dashdot) using the adjusted width */
        for (k = 0; k < penobj->numdashes; k++)
            adjdashes[k] = penobj->dashes[k] * adjwidth;
        cairo_set_dash(instdata->context, adjdashes, penobj->numdashes, 0.0);
        if ( cairo_status(instdata->context) != CAIRO_STATUS_SUCCESS ) {
            cairoCFerBind_setAntialias(self, antialias);
            strcpy(grdelerrmsg, "cairoCFerBind_drawRectangle: unexpected error, "
                                "problems setting pen dashes");
            return 0;
        }
        /* Assign the line cap and join styles */
        cairo_set_line_cap(instdata->context, penobj->captype);
        cairo_set_line_join(instdata->context, penobj->jointype);

        /* stroke and remove the path */
        cairo_stroke(instdata->context);
    }

    /* Restore the original antialiasing state */
    cairoCFerBind_setAntialias(self, antialias);

    instdata->somethingdrawn = 1;
    instdata->imagechanged = 1;
    return 1;
}

