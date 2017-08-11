/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "FerMem.h"

/*
 * Delete the given pen object.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_deletePen(CFerBind *self, grdelType pen)
{
    CCFBPen *penobj;

    /* Sanity checks */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deletePen: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    penobj = (CCFBPen *) pen;
    if ( penobj->id != CCFBPenId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deletePen: unexpected error, "
                            "pen is not CCFBPen struct");
        return 0;
    }

    /* Wipe the id to detect errors */
    penobj->id = NULL;

    /* Free the memory */
    FerMem_Free(pen, __FILE__, __LINE__);

    return 1;
}

