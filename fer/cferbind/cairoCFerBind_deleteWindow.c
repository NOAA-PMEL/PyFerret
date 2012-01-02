/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Deletes (frees) any allocated resources associated
 * with this instance of the bindings, and then the 
 * bindings itself.  After calling this function, the
 * bindings should no longer be used.
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

