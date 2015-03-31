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
    char *textString;
    double unitfactor;
    cairo_matrix_t fontsizematrix;
    cairo_matrix_t rotatematrix;
    cairo_font_options_t *fontoptions;
    cairo_scaled_font_t *scaledfont;

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
    if ( textlen < 1 ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawText: textlen is not positive");
        return 0;
    }

    /* Get the text as a null-terminated string */
    textString = (char *) PyMem_Malloc((textlen + 1) * sizeof(char));
    if ( textString == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawText: "
                            "out of memory for a copy of the text");
        return 0;
    }
    strncpy(textString, text, textlen);
    textString[textlen] = '\0';

    /* Conversion factor for those surfaces that expect points instead of pixels */
    if ( instdata->imageformat == CCFBIF_PNG ) {
        unitfactor = 1.0;
    }
    else {
        unitfactor = CCFB_POINTS_PER_PIXEL;
    }

    /* Create the scaled font to use */
    cairo_matrix_init_scale(&fontsizematrix, fontobj->fontsize, fontobj->fontsize);
    cairo_matrix_init_rotate(&rotatematrix, rotation * M_PI / 180.0);
    fontoptions = cairo_font_options_create();
    if ( cairo_font_options_status(fontoptions) != CAIRO_STATUS_SUCCESS ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawText: "
                            "out of memory for a font options structure");
        PyMem_Free(textString);
        return 0;
    }
    scaledfont = cairo_scaled_font_create(fontobj->fontface, &fontsizematrix, &rotatematrix, fontoptions);
    if ( cairo_scaled_font_status(scaledfont) != CAIRO_STATUS_SUCCESS ) {
        strcpy(grdelerrmsg, "cairoCFerBind_drawText: "
                            "out of memory for a font options structure");
        cairo_font_options_destroy(fontoptions);
        PyMem_Free(textString);
        return 0;
    }

    /* Assign the color for this text */
    if ( instdata->usealpha )
        cairo_set_source_rgba(instdata->context, colorobj->redfrac, 
                              colorobj->greenfrac, colorobj->bluefrac, colorobj->opaquefrac);
    else
        cairo_set_source_rgb(instdata->context, colorobj->redfrac, 
                             colorobj->greenfrac, colorobj->bluefrac);

    /* Move to the place to start drawing this text */
    cairo_new_path(instdata->context);
    cairo_move_to(instdata->context, startx * unitfactor, starty * unitfactor);

    /* Draw the text using this scaled font */
    cairo_save(instdata->context);
    cairo_set_scaled_font(instdata->context, scaledfont);
    cairo_show_text(instdata->context, textString);
    cairo_restore(instdata->context);

    cairo_scaled_font_destroy(scaledfont);
    cairo_font_options_destroy(fontoptions);
    PyMem_Free(textString);

    instdata->somethingdrawn = 1;
    instdata->imagechanged = 1;

    return 1;
}

