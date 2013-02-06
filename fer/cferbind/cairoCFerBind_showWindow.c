/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * "Show" this "Window".
 * In this case (Cairo), this function does nothing since there
 * is no displayed window associated with this engine.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_showWindow(CFerBind *self, int visible)
{
    /* Sanity check - this should NOT be called by the PyQtCairo engine */
    if ( self->enginename != CairoCFerBindName ) {
        strcpy(grdelerrmsg, "cairoCFerBind_showWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }

    return 1;
}

