/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Set the DPI for this "Window".  Adjusts the line width scaling factor.
 * Does not adjust window size since the should precede a window resize 
 * command.  Does not adjust font sizes as these are generated as needed.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_setWindowDpi(CFerBind *self, double newdpi)
{
    CairoCFerBindData *instdata;

    /* Sanity check  - allow PyQtCairoCFerBindName for internal use */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_setWindowDpi: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    instdata->widthfactor *= newdpi / instdata->pixelsperinch;
    instdata->pixelsperinch = newdpi;
    return 1;
}

