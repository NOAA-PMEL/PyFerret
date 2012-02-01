/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <cairo/cairo-pdf.h>
#include <cairo/cairo-ps.h>
#include <cairo/cairo-svg.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Creates the Cairo Surface and Context for this image.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_createSurface(CFerBind *self)
{
    CairoCFerBindData *instdata;
    double  width;
    double  height;
    char   *fmtname;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createSurface: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Create the surface if it does not exist */
    if ( instdata->surface == NULL ) {
        /*
         * Sanity check: the context includes the surface, but cairoCFerBind
         * also maintains a separate reference to the surface; thus, if there
         * is no surface, there should not be a context.
         */
        if ( instdata->context != NULL ) {
            strcpy(grdelerrmsg, "cairoCFerBind_createSurface: unexpected error, "
                                "NULL surface but non-NULL context");
            return 0;
        }
        /* Create the appropriate surface */
        switch( instdata->imageformat ) {
        case CCFBIF_PNG:
            /* Surface size is given in integer pixels */
            instdata->surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32,
                                      instdata->imagewidth, instdata->imageheight);
            instdata->usealpha = 1;
            /* Note that all surface values are initialized to zero (transparent) */
            fmtname = "PNG";
            break;
        case CCFBIF_PDF:
            /* Surface size is given in (floating-point) points */
            width = (double) instdata->imagewidth * CCFB_POINTS_PER_PIXEL;
            height = (double) instdata->imageheight * CCFB_POINTS_PER_PIXEL;
            instdata->surface = cairo_pdf_surface_create(instdata->imagename,
                                                         width, height);
            instdata->usealpha = 0;
            fmtname = "PDF";
            break;
        case CCFBIF_PS:
            /* Surface size is given in (floating-point) points */
            width = (double) instdata->imagewidth * CCFB_POINTS_PER_PIXEL;
            height = (double) instdata->imageheight * CCFB_POINTS_PER_PIXEL;
            if ( width > height ) {
                /*
                 * Landscape orientation
                 * Swap width and height and then translate and rotate (see
                 * below) per Cairo requirements.
                 */
                instdata->surface = cairo_ps_surface_create(instdata->imagename,
                                                            height, width);
            }
            else {
                /* Portrait orientation */
                instdata->surface = cairo_ps_surface_create(instdata->imagename,
                                                            width, height);
            }
            /* Do not use alpha channel - prevents embedded image */
            instdata->usealpha = 0;
            fmtname = "PS";
            break;
        case CCFBIF_SVG:
            /* Surface size is given in (floating-point) points */
            width = (double) instdata->imagewidth * CCFB_POINTS_PER_PIXEL;
            height = (double) instdata->imageheight * CCFB_POINTS_PER_PIXEL;
            instdata->surface = cairo_svg_surface_create(instdata->imagename,
                                                         width, height);
            instdata->usealpha = 1;
            fmtname = "SVG";
            break;
        default:
            sprintf(grdelerrmsg, "cairoCFerBind_createSurface: unexpected error, "
                                 "unknown imageformat %d", instdata->imageformat);
            return 0;
        }
        /* Check for failure to create the surface */
        if ( cairo_surface_status(instdata->surface) != CAIRO_STATUS_SUCCESS ) {
            sprintf(grdelerrmsg, "cairoCFerBind_createSurface: "
                                 "problems creating a %s surface", fmtname);
            cairo_surface_destroy(instdata->surface);
            instdata->surface = NULL;
            return 0;
        }
        /* set the resolution for fallback raster images in vector drawings */
        if ( instdata->imageformat != CCFBIF_PNG )
            cairo_surface_set_fallback_resolution(instdata->surface,
                              (double) CCFB_WINDOW_DPI, (double) CCFB_WINDOW_DPI);
    }

    /* Create the Context if it does not exist */
    if ( instdata->context == NULL ) {
        instdata->context = cairo_create(instdata->surface);
        if ( cairo_status(instdata->context) != CAIRO_STATUS_SUCCESS ) {
            strcpy(grdelerrmsg, "cairoCFerBind_createSurface: "
                                 "problems creating a context from a surface");
            cairo_destroy(instdata->context);
            instdata->context = NULL;
            cairo_surface_destroy(instdata->surface);
            instdata->surface = NULL;
            return 0;
        }
        /* Assign context values recorded in this instance */
        if ( instdata->antialias )
            cairo_set_antialias(instdata->context, CAIRO_ANTIALIAS_DEFAULT);
        else
            cairo_set_antialias(instdata->context, CAIRO_ANTIALIAS_NONE);
        /*
         * If landscape PostScript, translate and rotate the coordinate system
         * to correct for swapped width and height (per Cairo requirements).
         */
        if ( instdata->imageformat == CCFBIF_PS ) {
            width = (double) instdata->imagewidth * CCFB_POINTS_PER_PIXEL;
            height = (double) instdata->imageheight * CCFB_POINTS_PER_PIXEL;
            if ( width > height ) {
                /* surface was created with coordinates (0,0) to (height, width) */
                cairo_matrix_t transmat;

                /* Add a "comment" telling PostScript it is landscape */
                cairo_ps_surface_dsc_begin_page_setup(instdata->surface);
                cairo_ps_surface_dsc_comment(instdata->surface,
                                         "%%PageOrientation: Landscape");
                /* Move to the bottom left corner */
                cairo_translate(instdata->context, 0.0, width);
                /* Rotate 90 degrees clockwise */
                cairo_matrix_init(&transmat, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0);
                cairo_transform(instdata->context, &transmat);
                /*
                 * The transformed coordinate system goes from (0,0) at the top
                 * left corner to (width, height) at the bottom right corner.
                 */
            }
        }
        /* Set the appropriate clipping rectangle (if any) */
        if ( ! cairoCFerBind_clipView(self, instdata->clipit) ) {
            /* grdelerrmsg appropriately assigned */
            return 0;
        }
    }

    grdelerrmsg[0] = '\0';
    return 1;
}

