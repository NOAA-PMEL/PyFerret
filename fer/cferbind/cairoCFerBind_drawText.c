/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

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
    PangoLayout *layout;

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

    /* Conversion factor for those surfaces that expect points instead of pixels */
    if ( (instdata->imageformat == CCFBIF_PNG) ||
         (instdata->imageformat == CCFBIF_REC) ) {
        unitfactor = 1.0;
    }
    else {
        unitfactor = CCFB_POINTS_PER_PIXEL;
    }

    /* Assign the color for this text */
    if ( instdata->noalpha )
        cairo_set_source_rgb(instdata->context, colorobj->redfrac, 
                             colorobj->greenfrac, colorobj->bluefrac);
    else
        cairo_set_source_rgba(instdata->context, colorobj->redfrac, 
                              colorobj->greenfrac, colorobj->bluefrac, 
                              colorobj->opaquefrac);

    /* Move to the place to start drawing this text */
    cairo_move_to(instdata->context, startx * unitfactor, starty * unitfactor);

    /* If no text to draw, just return at this point */
    if ( textlen == 0 )
        return 1;

    /* Apply the rotation matrix and draw the text */
    cairo_save(instdata->context);
    cairo_rotate(instdata->context, rotation * M_PI / 180.0);
    layout = pango_cairo_create_layout(instdata->context);
    pango_layout_set_font_description(layout, fontobj->fontdesc);
    pango_layout_set_text(layout, text, textlen);
    pango_cairo_show_layout(instdata->context, layout);
    g_object_unref(layout);
    cairo_restore(instdata->context);

    instdata->somethingdrawn = 1;
    instdata->imagechanged = 1;

    return 1;
}

