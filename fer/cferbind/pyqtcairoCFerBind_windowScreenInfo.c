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
 * If an error occurs, grdelerrmsg is assigned an appropriate error 
 * message and zero is returned; otherwise one is returned.
 */
grdelBool pyqtcairoCFerBind_windowScreenInfo(CFerBind *self, 
                            float *dpix, float *dpiy,
                            int *screenwidth, int *screenheight)
{
    CairoCFerBindData *instdata;
    grdelBool success;

    /* Sanity check */
    if ( self->enginename != PyQtCairoCFerBindName ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_windowScreenInfo: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Get the values from viewer */
    success = grdelWindowScreenInfo(instdata->viewer, dpix, dpiy, 
                                    screenwidth, screenheight);
    if ( ! success ) {
       /* grdelerrmsg already assigned */
       return 0;
    }

    return 1;
}

