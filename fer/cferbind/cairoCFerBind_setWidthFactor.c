/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Set the scaling factor for line widths and symbol sizes
 * to convert from points (1/72 inches) to pixels, and to 
 * apply any additional width scaling specified by factor.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_setWidthFactor(CFerBind *self, double factor)
{
    CairoCFerBindData *instdata;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_setWidthFactor: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    if ( factor <= 0.0 ) {
        strcpy(grdelerrmsg, "cairoCFerBind_setWidthFactor: "
                            "scaling factor must be positive");
        return 0;
    }

    instdata = (CairoCFerBindData *) self->instancedata;
    instdata->widthfactor = CCFB_WINDOW_DPI * factor / 72.0;
    return 1;
}

