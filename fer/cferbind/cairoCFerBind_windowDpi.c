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

    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_windowDpi: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return NULL;
    }

    dpis[0] = CCFB_WINDOW_DPI;
    dpis[1] = CCFB_WINDOW_DPI;

    return dpis;
}

