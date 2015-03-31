/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Delete a font object for this "Window".
 *
 * Currently stubbed since it is currently not used by Ferret;
 * thus always fails.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_deleteFont(CFerBind *self, grdelType font)
{
    CCFBFont *fontobj;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deleteFont: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }

    fontobj = (CCFBFont *) font;
    if ( fontobj->id != CCFBFontId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deleteFont: unexpected error, "
                            "font is not CCFBFont struct");
        return 0;
    }

    if ( fontobj->fontface != NULL )
        cairo_font_face_destroy(fontobj->fontface);

    /* Wipe the id to detect errors */
    fontobj->id = NULL;

    /* Free the memory */
    PyMem_Free(font);

    return 1;
}

