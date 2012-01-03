/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Returns the horizontal and vertical resolution of this "Window"
 * in units of dots (pixels) per inch.  In this case (Cairo),
 * these are hard-coded values based on the surface type.
 *
 * Returns a pair of doubles (a static array in this function)
 * containing the DPIs if successful.  If an error occurs,
 * grdelerrmsg is assigned an appropriate error message and NULL
 * is returned.
 */
double * cairoCFerBind_windowDpi(CFerBind *self)
{
    static double dpis[2];

    CairoCFerBindData *instdata;

    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_windowDpi: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return NULL;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    switch( instdata->imageformat ) {
    case CCFBIF_PNG:
        dpis[0] = CCFB_RASTER_DPI;
        dpis[1] = CCFB_RASTER_DPI;
        break;
    case CCFBIF_PDF:
    case CCFBIF_EPS:
    case CCFBIF_SVG:
        dpis[0] = CCFB_VECTOR_DPI;
        dpis[1] = CCFB_VECTOR_DPI;
        break;
    default:
        sprintf(grdelerrmsg, "cairoCFerBind_windowDpi: unexpected error, "
                             "unknown imageformat %d", instdata->imageformat);
        return NULL;
    }

    return dpis;
}

