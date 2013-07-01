/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"

/*
 * Set the scaling factor for the displayed image.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool pyqtcairoCFerBind_scaleWindow(CFerBind *self, double scale)
{
    CairoCFerBindData *instdata;
    grdelBool success;

    /* Sanity check */
    if ( self->enginename != PyQtCairoCFerBindName ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_scaleWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Pass the scaling factor on the the image displayer */
    success = grdelWindowSetScale(instdata->viewer, scale);
    if ( ! success ) {
        /* grdelerrmsg already assigned */
        return 0;
    }

    return 1;
}

