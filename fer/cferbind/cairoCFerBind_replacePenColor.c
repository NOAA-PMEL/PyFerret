/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Replace the color in the given pen object with that in
 * the given color object.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_replacePenColor(CFerBind *self,
                                        grdelType pen, grdelType color)
{
    CCFBPen *penobj;
    CCFBColor *colorobj;

    /* Sanity checks */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_replacePenColor: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    penobj = (CCFBPen *) pen;
    if ( penobj->id != CCFBPenId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_replacePenColor: unexpected error, "
                            "pen is not CCFBPen struct");
        return 0;
    }
    colorobj = (CCFBColor *) color;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_replacePenColor: unexpected error, "
                            "color is not CCFBColor struct");
        return 0;
    }

    penobj->color = *colorobj;

    return 1;
}

