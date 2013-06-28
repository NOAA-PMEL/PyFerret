/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Returns information about the default screen (display) of
 * the window.  In this case (Cairo), these are hard-coded values.
 *
 * If an error occurs, grdelerrmsg is assigned an appropriate 
 * error message and zero is returned; otherwise one is returned.
 */
grdelBool cairoCFerBind_windowScreenInfo(CFerBind *self, 
                        float *dpix, float *dpiy,
                        int *screenwidth, int *screenheight)
{
    /* Sanity check - this should NOT be called by the PyQtCairo engine */
    if ( self->enginename != CairoCFerBindName ) {
        strcpy(grdelerrmsg, "cairoCFerBind_windowScreenInfo: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }

    *dpix = CCFB_WINDOW_DPI;
    *dpiy = CCFB_WINDOW_DPI;
    *screenwidth = 1920;
    *screenheight = 1080;
    return 1;
}

