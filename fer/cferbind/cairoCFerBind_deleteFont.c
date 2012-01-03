/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Delete a font object for this "Window".
 *
 * Currently stubbed since it is currently not used by Ferret;
 * thus always fails.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_deleteFont(CFerBind *self, grdelType font)
{
    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_deleteFont: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }

    /* TODO: implement */
    strcpy(grdelerrmsg, "cairoCFerBind_deleteFont: unexpected error, "
                        "stubbed function");
    return 0;
}

