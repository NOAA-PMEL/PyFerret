/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * "Clears the Window".
 *
 * This function destroys any existing surface and context if
 * anything has been drawn.  The clearing color is assigned to
 * the given color; however, the clearing color is only used
 * as the background color in raster images if transparency is
 * not desired.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_clearWindow(CFerBind *self, grdelType fillcolor)
{
    CairoCFerBindData *instdata;
    CCFBColor *colorobj;

    /* Sanity checks */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_clearWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    colorobj = (CCFBColor *) fillcolor;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_clearWindow: unexpected error, "
                            "fillcolor is not CCFBColor struct");
        return 0;
    }

    /* If something was drawn, delete the context and surface */
    if ( instdata->somethingdrawn ) {
        if ( instdata->context == NULL ) {
            strcpy(grdelerrmsg, "cairoCFerBind_clearWindow: unexpected error, "
                                "something drawn without a context");
            return 0;
        }
        if ( instdata->surface == NULL ) {
            strcpy(grdelerrmsg, "cairoCFerBind_clearWindow: unexpected error, "
                                "something drawn without a surface");
            return 0;
        }
        cairo_destroy(instdata->context);
        instdata->context = NULL;
        cairo_surface_destroy(instdata->surface);
        instdata->surface = NULL;
        instdata->somethingdrawn = 0;
        instdata->imagechanged = 1;
    }

    /* Copy the given color structure values to lastclearcolor */
    instdata->lastclearcolor = *colorobj;

    return 1;
}

