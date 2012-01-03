/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * "Clears the Window".
 * In this case (Cairo), this function destroys any existing
 * surface and context if anything has been drawn.  The clearing
 * color is assigned to the given color; however, the clearing
 * color is only used as the background color in raster images
 * (PNG) if transparency if not desired.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_clearWindow(CFerBind *self, grdelType fillcolor)
{
    CairoCFerBindData *instdata;
    CCFBColor *colorobj;

    /* Sanity checks */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_clearWindow: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    colorobj = (CCFBColor *) fillcolor;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_clearWindow: unexpected error, "
                            "fillcolor is not CCFBColor struct");
        return 0;
    }

    /* If something was drawn, delete any existing context and surface */
    if ( instdata->somethingdrawn ) {
        if ( instdata->context != NULL ) {
            cairo_destroy(instdata->context);
            instdata->context = NULL;
        }
        if ( instdata->surface != NULL ) {
            cairo_surface_destroy(instdata->surface);
            instdata->surface = NULL;
        }
        instdata->somethingdrawn = 0;
    }

    /* Copy the given color structure values to lastclearcolor */
    instdata->lastclearcolor = *colorobj;

    return 1;
}

