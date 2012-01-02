/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Delete the given color object
 *
 * Returns zero if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and one is returned.
 */
grdelBool cairoCFerBind_deleteColor(CFerBind *self, grdelType color)
{
    CCFBColor *colorobj;

    /* Sanity checks */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_deleteColor: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
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

