/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * "Show" this "Window".
 * In this case (Cairo), this function does nothing since there
 * is no displayed window associated with this engine.
 *
 * One is returned on success.  If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_showWindow(CFerBind *self, int visible)
{
    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_endView: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }

    return 1;
}

