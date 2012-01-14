/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Create a font object for this "Window".
 *
 * Currently stubbed since it is currently not used by Ferret;
 * thus always fails.
 *
 * Returns a font object if successful.   If an error occurs,
 * grdelerrmsg is assigned an appropriate error message and NULL
 * is returned.
 */
grdelType cairoCFerBind_createFont(CFerBind *self, const char *familyname, int namelen,
                        double fontsize, int italic, int bold, int underlined)
{
    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createFont: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }

    /* TODO: implement */
    strcpy(grdelerrmsg, "cairoCFerBind_createFont: unexpected error, "
                        "stubbed function");
    return NULL;
}

