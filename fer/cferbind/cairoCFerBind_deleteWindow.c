/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

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

    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_deleteWindow: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Delete any existing context and surface */
    if ( instdata->context != NULL ) {
        cairo_destroy(instdata->context);
        instdata->context = NULL;
    }
    if ( instdata->surface != NULL ) {
        cairo_surface_destroy(instdata->surface);
        instdata->surface = NULL;
    }
    PyMem_Free(self->instancedata);
    self->instancedata = NULL;
    PyMem_Free(self);
    return 1;
}

