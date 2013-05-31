/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"

/*
 * Redraws the displayed scene with the updated background color.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool pyqtcairoCFerBind_redrawWindow(CFerBind *self, grdelType fillcolor)
{
    CairoCFerBindData *instdata;
    CCFBColor *colorobj;
    grdelType  viewercolor;
    grdelBool  success;

    /* Sanity checks */
    if ( self->enginename != PyQtCairoCFerBindName ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_redrawWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    colorobj = (CCFBColor *) fillcolor;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_redrawWindow: unexpected error, "
                            "fillcolor is not CCFBColor struct");
        return 0;
    }

    /* Update the clearing color for cairo */
    success = cairoCFerBind_redrawWindow(self, fillcolor);
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

    /* Tell the viewer to redraw the current scene using this background color */
    success = grdelWindowRedraw(instdata->viewer, viewercolor);
    if ( ! success ) {
        /* delete the viewer color created here */
        grdelColorDelete(viewercolor);
        /* grdelerrmsg assigned by grdelWindowClear failure */
        return 0;
    }

    /* Delete the viewer color object created here */
    success = grdelColorDelete(viewercolor);
    if ( ! success ) {
       /* grdelerrmsg already assigned */
       return 0;
    }

    /*
     * No changes were made to either the Cairo image or the image
     * in the viewer window, so do not mess with instdata->imagechanged
     */

    return 1;
}

