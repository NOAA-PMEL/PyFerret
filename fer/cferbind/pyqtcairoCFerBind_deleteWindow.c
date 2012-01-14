/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"

/*
 * Deletes (frees) any allocated resources associated with this
 * "Window" (bindings instance), and then deletes (frees) the
 * bindings instance.  After calling this function, this bindings
 * instance should no longer be used.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool pyqtcairoCFerBind_deleteWindow(CFerBind *self)
{
    CairoCFerBindData *instdata;
    grdelBool success;

    /* Sanity check */
    if ( self->enginename != PyQtCairoCFerBindName ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_deleteWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /*
     * First shut down the viewer and delete its bindings
     * (contained in the instance data under these bindings).
     */
    success = grdelWindowDelete(instdata->viewer);
    if ( ! success ) {
        /* grdelerrmsg already defined */
        return 0;
    }

    /*
     * Then delete the cairo context, cairo surface, and
     * these bindings.
     */
    success = cairoCFerBind_deleteWindow(self);
    if ( ! success ) {
        /* grdelerrmsg already defined */
        return 0;
    }

    return 1;
}

