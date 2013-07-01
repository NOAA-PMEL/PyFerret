/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * "Sets the image scaling factor for a Window".
 *
 * This function does nothing after checking that self is valid,
 * as the image is not displayed.
 * 
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_scaleWindow(CFerBind *self, double scale)
{
    /* Sanity checks */
    if ( self->enginename != CairoCFerBindName ) {
        strcpy(grdelerrmsg, "cairoCFerBind_scaleWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }

    return 1;
}

