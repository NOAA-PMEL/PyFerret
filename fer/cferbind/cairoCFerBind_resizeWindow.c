/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Resizes the "Window" to the given width and height.  These
 * values are assumed to be in units of pixels.  In this case
 * (Cairo), if the size changes, this function destroys any
 * existing surface and context; thus clearing the image.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_resizeWindow(CFerBind *self, double width, double height)
{
    CairoCFerBindData *instdata;
    int newwidth;
    int newheight;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_resizeWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
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
    if ( (instdata->imagewidth == newwidth) && (instdata->imageheight == newheight) )
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
        instdata->imagechanged = 1;
    }
    instdata->somethingdrawn = 0;

    return 1;
}

