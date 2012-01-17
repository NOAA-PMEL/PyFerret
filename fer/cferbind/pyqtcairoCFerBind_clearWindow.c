/* Python.h should always be first */
#include <Python.h>
#include <string.h>
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
    grdelBool  success;

    /* Sanity checks */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
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

    /* Tell the viewer to clear the displayed scene */
    success = grdelWindowClear(instdata->viewer, viewercolor);
    if ( ! success ) {
        char myerrmsg[2048];
        /* copy the error message (grdelColorDelete will clear it) */
        strcpy(myerrmsg, grdelerrmsg);
        /* delete the viewer color created hete */
        grdelColorDelete(viewercolor);
        /* return the error message to grdelerrmsg */
        strcpy(grdelerrmsg, myerrmsg);
    }

    /* Delete the viewer color object created here */
    success = grdelColorDelete(viewercolor);
    if ( ! success ) {
       /* grdelerrmsg already assigned */
       return 0;
    }

    /*
     * The viewer window was cleared, and the Cairo context and
     * surface were deleted, so the images match.
     */
    instdata->imagechanged = 0;

    return 1;
}

