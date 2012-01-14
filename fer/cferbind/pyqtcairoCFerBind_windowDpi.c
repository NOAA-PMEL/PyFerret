/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"

/*
 * Returns the horizontal and vertical resolution of this "Window"
 * in units of dots (pixels) per inch.
 *
 * Returns a pair of doubles (a static array in this function)
 * containing the DPIs if successful.  If an error occurs,
 * grdelerrmsg is assigned an appropriate error message and NULL
 * is returned.
 */
double * pyqtcairoCFerBind_windowDpi(CFerBind *self)
{
    static double dpis[2];
    CairoCFerBindData *instdata;
    grdelBool success;
    float dpix, dpiy;

    /* Sanity check */
    if ( self->enginename != PyQtCairoCFerBindName ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_windowDpi: unexpected error, "
                            "self is not a valid CFerBind struct");
        return NULL;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Get the DPIs from viewer */
    success = grdelWindowDpi(instdata->viewer, &dpix, &dpiy);
    if ( ! success ) {
       /* grdelerrmsg already assigned */
       return NULL;
    }

    dpis[0] = (double) dpix;
    dpis[1] = (double) dpiy;
    return dpis;
}

