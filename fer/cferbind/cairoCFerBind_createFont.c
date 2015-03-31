/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/* Instantiate the global value */
const char *CCFBFontId = "CCFBFontId";

/*
 * Create a font object for this "Window".
 *
 * Currently stubbed since it is currently not used by Ferret;
 * thus always fails.
 *
 * Returns a font object if successful.   If an error occurs,
 * grdelerrmsg is assigned an appropriate error message and NULL
 * is returned.
 */
grdelType cairoCFerBind_createFont(CFerBind *self, const char *familyname, int namelen,
                        double fontsize, int italic, int bold, int underlined)
{
    char *family;
    cairo_font_slant_t slant;
    cairo_font_weight_t weight;
    CCFBFont *fontobj;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createFont: unexpected error, "
                            "self is not a valid CFerBind struct");
        return NULL;
    }
    if ( (familyname == NULL) || (namelen < 0) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createFont: familyname is not given");
        return NULL;
    }
    if ( fontsize <= 0.0 ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createFont: invalid font size given");
        return NULL;
    }

    fontobj = (CCFBFont *) PyMem_Malloc(sizeof(CCFBFont));
    if ( fontobj == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createFont: "
                            "out of memory for a CCFBFont structure");
        return NULL;
    }
    fontobj->id = CCFBFontId;

    family = (char *) PyMem_Malloc(namelen+1);
    if ( family == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createFont: "
                            "out of memory for a copy of the familyname string");
        PyMem_Free(fontobj);
        return NULL;
    }
    strncpy(family, familyname, namelen);
    family[namelen] = '\0';

    if ( italic == 0 )
        slant = CAIRO_FONT_SLANT_NORMAL;
    else
        slant = CAIRO_FONT_SLANT_ITALIC;
    if ( bold == 0 )
        weight = CAIRO_FONT_WEIGHT_NORMAL;
    else
        weight = CAIRO_FONT_WEIGHT_BOLD;

    fontobj->fontface = cairo_toy_font_face_create(family, slant, weight);

    PyMem_Free(family);

    fontobj->fontsize = fontsize;
    fontobj->underline = underlined;

    return fontobj;
}

