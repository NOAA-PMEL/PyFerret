/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Return the text size if drawn to this "Window" using the given font.
 * The value returned at widthptr is amount to advance in X direction 
 * to draw any subsequent text after this text (not the width of the
 * text glyphs as drawn).  The value returned at heightptr is the height 
 * of the text glyphs as drawn.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_textSize(CFerBind *self, const char *text, int textlen,
                                 grdelType font, double *widthptr, double *heightptr)
{
    CairoCFerBindData *instdata;
    CCFBFont  *fontobj;
    char      *textString;
    cairo_text_extents_t extents;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_textSize: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    if ( instdata->context == NULL ) {
        /* Create the Cairo Surface and Context if they do not exist */
        if ( ! cairoCFerBind_createSurface(self) ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    fontobj = (CCFBFont *) font;
    if ( fontobj->id != CCFBFontId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_textSize: unexpected error, "
                            "font is not CCFBFont struct");
        return 0;
    }
    if ( textlen < 1 ) {
        strcpy(grdelerrmsg, "cairoCFerBind_textSize: textlen is not positive");
        return 0;
    }

    /* Get the text as a null-terminated string */
    textString = (char *) PyMem_Malloc((textlen + 1) * sizeof(char));
    if ( textString == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_textSize: "
                            "out of memory for a copy of the text");
        return 0;
    }
    strncpy(textString, text, textlen);
    textString[textlen] = '\0';

    /* Get the extents and advance values of this text if drawn using the current font */
    cairo_save(instdata->context);
    cairo_set_font_face(instdata->context, fontobj->fontface);
    cairo_set_font_size(instdata->context, fontobj->fontsize);
    cairo_text_extents(instdata->context, textString, &extents);
    cairo_restore(instdata->context);

    /* Assign the X advance value (and not the width) to *widthptr */
    *widthptr = extents.x_advance;
    *heightptr = extents.height;

    return 0;
}

