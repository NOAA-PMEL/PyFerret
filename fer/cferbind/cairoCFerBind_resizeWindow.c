/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Resizes the "Window" to the given width and height.  These
 * values are assumed to be in units of pixels.  In this case
 * (Cairo), if the size changes, this function destroys any
 * existing surface and context (thus also clearing the image).
 *
 * Returns zero if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and one is returned.
 */
grdelBool cairoCFerBind_resizeWindow(CFerBind *self, double width, double height)
{
    CairoCFerBindData *instdata;
    int newwidth;
    int newheight;

    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_resizeWindow: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Round width and height to the nearest integer value */
    newwidth = (int) (width + 0.5);
    newheight = (int) (height + 0.5);

    /* Check the new width and height */
    if ( (newwidth < instdata->minsize) || (newheight < instdata->minsize) ) {
        sprintf(grdelerrmsg, "cairoCFerBind_resizeWindow: size too small, "
                             "width (%d) and height (%d) cannot be less "
                             "than %d", newwidth, newheight, instdata->minsize);
        return 0;
    }

    /* If the size is unchanged, nothing to do */
    if ( (instdata->imagewidth == newwidth) && (instdata->imageheight == height) )
        return 1;

    /* Assign the new size */
    instdata->imagewidth = newwidth;
    instdata->imageheight = newheight;

    /* Delete any existing context and surface which uses the old size */
    if ( instdata->context != NULL ) {
        cairo_destroy(instdata->context);
        instdata->context = NULL;
    }
    if ( instdata->surface != NULL ) {
        cairo_surface_destroy(instdata->surface);
        instdata->surface = NULL;
    }
    instdata->somethingdrawn = 0;

    return 1;
}

