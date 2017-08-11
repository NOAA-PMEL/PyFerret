/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "FerMem.h"

/*
 * Deletes (frees) any allocated resources associated with this
 * "Window" (bindings instance), and then deletes (frees) the
 * bindings instance.  After calling this function, this bindings
 * instance should no longer be used.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_deleteWindow(CFerBind *self)
{
    CairoCFerBindData *instdata;
    CCFBPicture *delpic;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deleteWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Delete any existing context and surface */
    if ( instdata->context != NULL ) {
        /*
         * Explicitly call cairo_show_page before destroying the context.
         * Cairo 1.2 (but not 1.4 or later) requires this call.
         */
        cairo_show_page(instdata->context);
        cairo_destroy(instdata->context);
        instdata->context = NULL;
    }
    if ( instdata->surface != NULL ) {
        /* Explicitly finish the surface just to be safe */
        cairo_surface_finish(instdata->surface);
        cairo_surface_destroy(instdata->surface);
        instdata->surface = NULL;
    }
    /* Delete any stored pictures */
    while ( instdata->firstpic != NULL ) {
        delpic = instdata->firstpic;
        instdata->firstpic = delpic->next;
        cairo_surface_finish(delpic->surface);
        cairo_surface_destroy(delpic->surface);
        FerMem_Free(delpic, __FILE__, __LINE__);
    }
    instdata->lastpic = NULL;

    FerMem_Free(self->instancedata, __FILE__, __LINE__);
    self->instancedata = NULL;
    FerMem_Free(self, __FILE__, __LINE__);
    return 1;
}

