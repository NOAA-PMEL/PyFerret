/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Delete the given color object.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_deleteColor(CFerBind *self, grdelType color)
{
    CCFBColor *colorobj;

    /* Sanity checks */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deleteColor: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    colorobj = (CCFBColor *) color;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deleteColor: unexpected error, "
                            "color is not CCFBColor struct");
        return 0;
    }

    /* Wipe the id to detect errors */
    colorobj->id = NULL;

    /* Free the memory */
    PyMem_Free(color);

    return 1;
}

