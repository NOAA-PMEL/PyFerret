/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "utf8str.h"

/*
 * Draw text to this "Window".
 * rotation is in degrees clockwise from horizontal.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_drawText(CFerBind *self, const char *text, int textlen,
                                 double startx, double starty, grdelType font,
                                 grdelType color, double rotation)
{
    CairoCFerBindData *instdata;
    CCFBFont *fontobj;
    CCFBColor *colorobj;
    double unitfactor;
    char *utf8str;
    int utf8strlen;
#ifdef USEPANGOCAIRO
    PangoLayout *layout;
#endif
    cairo_status_t result;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawText: unexpected error, "
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
        strcpy(grdelerrmsg, "cairoCFerBind_drawText: unexpected error, "
                            "font is not CCFBFont struct");
        return 0;
    }
    colorobj = (CCFBColor *) color;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawText: unexpected error, "
                            "color is not CCFBColor struct");
        return 0;
    }
    if ( textlen < 0 ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawText: textlen is not positive");
        return 0;
    }

    /* Assign the color for this text */
    if ( instdata->noalpha )
        cairo_set_source_rgb(instdata->context, colorobj->redfrac, 
                             colorobj->greenfrac, colorobj->bluefrac);
    else
        cairo_set_source_rgba(instdata->context, colorobj->redfrac, 
                              colorobj->greenfrac, colorobj->bluefrac, 
                              colorobj->opaquefrac);

    /* Conversion factor for those surfaces that expect points instead of pixels */
    if ( instdata->imageformat == CCFBIF_PNG ) {
        unitfactor = 1.0;
    }
    else {
        unitfactor = 72.0 / instdata->pixelsperinch;
    }

    /* Move to the place to start drawing this text */
    cairo_move_to(instdata->context, startx * unitfactor, starty * unitfactor);

    /* If no text to draw, just return at this point */
    if ( textlen == 0 )
        return 1;

    /* Convert to a null-terminated UTF-8 string (convert characters > 0x7F) */
    utf8str = (char *) PyMem_Malloc((2*textlen + 1) * sizeof(char));
    if ( utf8str == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawText: "
                            "out of memory for a UTF-8 copy of the text string");
        return 0;
    }
    text_to_utf8_(text, &textlen, utf8str, &utf8strlen);

    /* draw the text */
    cairo_save(instdata->context);
    cairo_rotate(instdata->context, rotation * M_PI / 180.0);
#ifdef USEPANGOCAIRO
    layout = pango_cairo_create_layout(instdata->context);
    pango_layout_set_font_description(layout, fontobj->fontdesc);
    pango_layout_set_text(layout, text, textlen);
    pango_cairo_show_layout(instdata->context, layout);
    g_object_unref(layout);
#else
    cairo_set_font_face(instdata->context, fontobj->fontface);
    /* fontsize has already been adjusted appropriately for this surface */
    cairo_set_font_size(instdata->context, fontobj->fontsize);
    cairo_show_text(instdata->context, utf8str);
#endif
    result = cairo_status(instdata->context);
    cairo_restore(instdata->context);

    PyMem_Free(utf8str);
    instdata->somethingdrawn = 1;
    instdata->imagechanged = 1;

    if ( result != CAIRO_STATUS_SUCCESS ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawText: "
                            "drawing the text was not successful");
        return 0;
    }

    return 1;
}

