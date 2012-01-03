/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Create a symbol object for this "Window".
 *
 * Currently stubbed since it is currently not used by Ferret;
 * thus always fails.
 *
 * Returns a sybmol object if successful.   If an error occurs,
 * grdelerrmsg is assigned an appropriate error message and NULL
 * is returned.
 */
grdelType cairoCFerBind_createSymbol(CFerBind *self, char *symbolname, int namelen)
{
    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_createSymbol: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return NULL;
    }

    /* TODO: implement */
    strcpy(grdelerrmsg, "cairoCFerBind_createSymbol: unexpected error, "
                        "stubbed function");
    return NULL;
}

