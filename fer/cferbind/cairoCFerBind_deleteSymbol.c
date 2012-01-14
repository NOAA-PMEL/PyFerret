/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Delete a symbol object for this "Window".
 *
 * Currently stubbed since it is currently not used by Ferret;
 * thus always fails.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_deleteSymbol(CFerBind *self, grdelType symbol)
{
    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deleteSymbol: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }

    /*
     * A symbol "object" is just a character value cast as a pointer type;
     * thus nothing needs to be done to delete it.
     */
    grdelerrmsg[0] = '\0';
    return 1;
}

