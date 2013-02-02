/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Turns on or off clipping of subsequent drawing to the current
 * view rectangle.
 *
 * Arguments:
 *     clipit - clip drawing to the current view rectangle?
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_clipView(CFerBind *self, int clipit)
{
    CairoCFerBindData *instdata;
    double left, bottom, right, top;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_clipView: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    instdata->clipit = clipit;

    if ( instdata->context == NULL ) {
        /* No context yet, so done */
        return 1;
    }

    /* Clear any clipping rectangle that may be present in the context */
    cairo_reset_clip(instdata->context);

    /* If no clipping desired, done */
    if ( ! clipit )
        return 1;

    /*
     * Convert the view fractions to positions on the surface.
     * No need to revalidate these positions since they are
     * floating-point values.
     */
    left = instdata->fracsides.left * instdata->imagewidth;
    right = instdata->fracsides.right * instdata->imagewidth;
    top = instdata->fracsides.top * instdata->imageheight;
    bottom = instdata->fracsides.bottom * instdata->imageheight;
    if ( instdata->imageformat != CCFBIF_PNG ) {
        left   *= CCFB_POINTS_PER_PIXEL;
        right  *= CCFB_POINTS_PER_PIXEL;
        top    *= CCFB_POINTS_PER_PIXEL;
        bottom *= CCFB_POINTS_PER_PIXEL;
    }

    /* Create the clipping rectangle path */
    cairo_new_path(instdata->context);
    cairo_rectangle(instdata->context, left, top, right - left, bottom - top);

    /* Assign the clipping rectangle, removing the above path */
    cairo_clip(instdata->context);

    return 1;
}

