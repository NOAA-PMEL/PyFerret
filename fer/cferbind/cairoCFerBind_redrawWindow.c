/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * "Redraws the Window with a different background color".
 *
 * This function only changes the clearing color to the given
 * background (fill) color.  Since this clearing color is not 
 * used until the image is saved, nothing more needs to be done.
 * 
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_redrawWindow(CFerBind *self, grdelType fillcolor)
{
    CairoCFerBindData *instdata;
    CCFBColor *colorobj;

    /* Sanity checks */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_redrawWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    colorobj = (CCFBColor *) fillcolor;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_redrawWindow: unexpected error, "
                            "fillcolor is not CCFBColor struct");
        return 0;
    }

    /* Copy the given color structure values to lastclearcolor */
    instdata->lastclearcolor = *colorobj;

    return 1;
}

