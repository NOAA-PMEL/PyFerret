/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"

/*
 * Resizes the "Window" to the given width and height.  These
 * values are assumed to be in units of pixels.  If the size
 * changes, this function clears the displayed scene and destroys
 * any existing surface and context; thus clearing the image.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool pyqtcairoCFerBind_resizeWindow(CFerBind *self, double width, double height)
{
    CairoCFerBindData *instdata;
    int       newwidth;
    int       newheight;
    grdelBool success;

    /* Sanity check */
    if ( self->enginename != PyQtCairoCFerBindName ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_resizeWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Round width and height to the nearest integer value */
    newwidth = (int) (width + 0.5);
    newheight = (int) (height + 0.5);

    /* Check the new width and height */
    if ( (newwidth < instdata->minsize) || (newheight < instdata->minsize) ) {
        sprintf(grdelerrmsg, "pyqtcairoCFerBind_resizeWindow: size too small, "
                             "width (%d) and height (%d) cannot be less "
                             "than %d", newwidth, newheight, instdata->minsize);
        return 0;
    }

    /* If the size is unchanged, nothing to do */
    if ( (instdata->imagewidth == newwidth) && (instdata->imageheight == newheight) )
        return 1;

    /* Set the new size for the cairo image */
    success = cairoCFerBind_resizeWindow(self, (double) newwidth, (double) newheight);
    if ( ! success ) {
        /* grdleerrmsg is already assigned */
        return 0;
    }

    /* Tell the viewer of the new size */
    success = grdelWindowSetSize(instdata->viewer, (double) newwidth, (double) newheight);
    if ( ! success ) {
        /* grdleerrmsg is already assigned */
        return 0;
    }
    /*
     * This assumes that the resize in the viewer cleared the window.
     * The resize in Cairo deleted the context and surface.
     */
    instdata->imagechanged = 0;

    return 1;
}

