/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Delete the given brush object.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_deleteBrush(CFerBind *self, grdelType brush)
{
    CCFBBrush *brushobj;

    /* Sanity checks */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_deleteBrush: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }
    brushobj = (CCFBBrush *) brush;
    if ( brushobj->id != CCFBBrushId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deleteBrush: unexpected error, "
                            "brush is not CCFBBrush struct");
        return 0;
    }

    /* Destroy any pattern given in this brush */
    if ( brushobj->pattern != NULL )
        cairo_pattern_destroy(brushobj->pattern);

    /* Wipe the id to detect errors */
    brushobj->id = NULL;

    /* Free the memory */
    PyMem_Free(brush);

    return 1;
}

