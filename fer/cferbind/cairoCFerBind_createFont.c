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
 * Returns a font object if successful.   If an error occurs,
 * grdelerrmsg is assigned an appropriate error message and NULL
 * is returned.
 */
grdelType cairoCFerBind_createFont(CFerBind *self, const char *familyname, int namelen,
                        double fontsize, int italic, int bold, int underlined)
{
    char *family;
    CCFBFont *fontobj;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createFont: unexpected error, "
                            "self is not a valid CFerBind struct");
        return NULL;
    }
    if ( familyname == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createFont: familyname is not given");
        return NULL;
    }
    if ( namelen < 0 ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createFont: invalid familyname length given");
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
    fontobj->fontdesc = pango_font_description_new();

    family = (char *) PyMem_Malloc(namelen+1);
    if ( family == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createFont: "
                            "out of memory for a copy of the familyname string");
        PyMem_Free(fontobj);
        return NULL;
    }
    strncpy(family, familyname, namelen);
    family[namelen] = '\0';
    pango_font_description_set_family(fontobj->fontdesc, family);
    PyMem_Free(family);

    /* If italic not supported, will switch to oblique */
    if ( italic == 0 )
        pango_font_description_set_style(fontobj->fontdesc, PANGO_STYLE_NORMAL);
    else
        pango_font_description_set_style(fontobj->fontdesc, PANGO_STYLE_ITALIC);
    /* Many weight options */
    if ( bold == 0 )
        pango_font_description_set_weight(fontobj->fontdesc, PANGO_WEIGHT_NORMAL);
    else
        pango_font_description_set_weight(fontobj->fontdesc, PANGO_WEIGHT_BOLD);
    /* Other variant option is PANGO_VARIANT_SMALL_CAPS */
    pango_font_description_set_variant(fontobj->fontdesc, PANGO_VARIANT_NORMAL);
    /* Many stretch options */
    pango_font_description_set_stretch(fontobj->fontdesc, PANGO_STRETCH_NORMAL);
    /* Set the size in points scaled by PANGO_SCALE (1024) */
    pango_font_description_set_size(fontobj->fontdesc, (int) (fontsize * PANGO_SCALE + 0.5));

    fontobj->underline = underlined;

    fontobj->id = CCFBFontId;
    return fontobj;
}

