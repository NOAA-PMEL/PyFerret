/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Returns information about the default screen (display) of the window. 
 *
 * If an error occurs, grdelerrmsg is assigned an appropriate 
 * error message and zero is returned; otherwise one is returned.
 */
grdelBool cairoCFerBind_windowScreenInfo(CFerBind *self, 
                        float *dpix, float *dpiy,
                        int *screenwidth, int *screenheight)
{
    CairoCFerBindData *instdata;

    /* Sanity check - this should NOT be called by the PyQtCairo engine */
    if ( self->enginename != CairoCFerBindName ) {
        strcpy(grdelerrmsg, "cairoCFerBind_windowScreenInfo: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    *dpix = (float) instdata->pixelsperinch;
    *dpiy = (float) instdata->pixelsperinch;
    *screenwidth = (int) (20 * instdata->pixelsperinch);
    *screenheight = (int) (12 * instdata->pixelsperinch);
    return 1;
}

