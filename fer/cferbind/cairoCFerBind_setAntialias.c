/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Turns on or off anti-aliasing on non-text graphics.
 *
 * Arguments:
 *     antialias - if zero, do not use anti-aliasing;
 *                 otherwise, use default anti-aliasing
 *
 * One is returned on success.  If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_setAntialias(CFerBind *self, int antialias)
{
    CairoCFerBindData *instdata;

    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_setAntialias: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Record the desired antialias setting */
    instdata->antialias = antialias;

    /* Also set the antialiasing value in the context, if it exists */
    if ( instdata->context != NULL ) {
        if ( antialias )
            cairo_set_antialias(instdata->context, CAIRO_ANTIALIAS_DEFAULT);
        else
            cairo_set_antialias(instdata->context, CAIRO_ANTIALIAS_NONE);
    }

    return 1;
}

