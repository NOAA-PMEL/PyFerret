/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "ferret.h"
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"

/*
 * Clears the displayed scene and the drawn image.
 *
 * This function destroys any existing surface and context if
 * anything has been drawn.  The viewer is also cleared using
 * the given color.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool pyqtcairoCFerBind_clearWindow(CFerBind *self, grdelType fillcolor)
{
    CairoCFerBindData *instdata;
    CCFBColor *colorobj;
    grdelType  viewercolor;
    int        inanimation;
    grdelBool  success;

    /* Sanity checks */
    if ( self->enginename != PyQtCairoCFerBindName ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_clearWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    colorobj = (CCFBColor *) fillcolor;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_clearWindow: unexpected error, "
                            "fillcolor is not CCFBColor struct");
        return 0;
    }

    /* If something was drawn, delete the context and surface */
    success = cairoCFerBind_clearWindow(self, fillcolor);
    if ( ! success ) {
        /* grdelerrmsg already assigned */
        return 0;
    }

    /* Create the color object for the viewer */
    viewercolor = grdelColor(instdata->viewer, colorobj->redfrac,
                             colorobj->greenfrac, colorobj->bluefrac,
                             colorobj->opaquefrac);
    if ( viewercolor == NULL ) {
       /* grdelerrmsg already assigned */
       return 0;
    }

    /* Only clear the displayed image if this is not in an animation */
    FORTRAN(fgd_getanimate)(&inanimation);
    if ( ! inanimation ) {
        /* Tell the viewer to clear the displayed scene */
        success = grdelWindowClear(instdata->viewer, viewercolor);
        if ( ! success ) {
            /* delete the viewer color created here */
            grdelColorDelete(viewercolor);
            /* grdelerrmsg assigned by grdelWindowClear failure */
            return 0;
        }
    }

    /* Delete the viewer color object created here */
    success = grdelColorDelete(viewercolor);
    if ( ! success ) {
       /* grdelerrmsg already assigned */
       return 0;
    }

    if ( ! inanimation ) {
        /*
         * The viewer window was cleared, and the Cairo context and
         * surface were deleted, so the images match.
         */
        instdata->imagechanged = 0;
    }
    else {
        /*
         * The viewer window was not cleared, but the Cairo context
         * and surface were deleted, so the images probably do not match.
         */
        instdata->imagechanged = 1;
    }

    return 1;
}

