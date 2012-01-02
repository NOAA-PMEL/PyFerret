/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Ends a "View" for this "Window".
 * In this case (Cairo), this function currently does nothing.
 *
 * One is returned on success.  If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_endView(CFerBind *self)
{
    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_endView: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }

    return 1;
}

