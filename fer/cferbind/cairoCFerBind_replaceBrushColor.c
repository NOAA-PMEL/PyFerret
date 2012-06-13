/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Replace the color in the given brush object with that in
 * the given color object.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_replaceBrushColor(CFerBind *self,
                                          grdelType brush, grdelType color)
{
    CCFBBrush *brushobj;
    CCFBColor *colorobj;

    /* Sanity checks */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_replaceBrushColor: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    brushobj = (CCFBBrush *) brush;
    if ( brushobj->id != CCFBBrushId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_replaceBrushColor: unexpected error, "
                            "brush is not CCFBBrush struct");
        return 0;
    }
    colorobj = (CCFBColor *) color;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_replaceBrushColor: unexpected error, "
                            "color is not CCFBColor struct");
        return 0;
    }

    brushobj->color = *colorobj;

    return 1;
}

