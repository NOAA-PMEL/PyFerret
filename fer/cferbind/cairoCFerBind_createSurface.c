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
    char              *fmtname;
    double            width;
    double            height;
    cairo_rectangle_t extents;

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
            if ( instdata->noalpha )
                instdata->surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24,
                                      instdata->imagewidth, instdata->imageheight);
            else
                instdata->surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32,
                                      instdata->imagewidth, instdata->imageheight);
            /* Note that all surface values are initialized to zero (transparent) */
            fmtname = "PNG";
            break;
        case CCFBIF_PDF:
            /* Surface size is given in (floating-point) points */
            width = instdata->imagewidth * 72.0 / instdata->pixelsperinch;
            height = instdata->imageheight * 72.0 / instdata->pixelsperinch;
            instdata->surface = cairo_pdf_surface_create(instdata->imagename, width, height);
            /* Never use the alpha channel to avoid embedded image */
            instdata->noalpha = 1;
            fmtname = "PDF";
            break;
        case CCFBIF_PS:
            /* Surface size is given in (floating-point) points */
            width = instdata->imagewidth * 72.0 / instdata->pixelsperinch;
            height = instdata->imageheight * 72.0 / instdata->pixelsperinch;
            if ( width > height ) {
                /*
                 * Landscape orientation
                 * Swap width and height and then translate and rotate (see
                 * below) per Cairo requirements.
                 */
                instdata->surface = cairo_ps_surface_create(instdata->imagename, height, width);
            }
            else {
                /* Portrait orientation */
                instdata->surface = cairo_ps_surface_create(instdata->imagename, width, height);
            }
            /* Never use the alpha channel to avoid embedded image */
            instdata->noalpha = 1;
            fmtname = "PS";
            break;
        case CCFBIF_SVG:
            /* Surface size is given in (floating-point) points */
            width = instdata->imagewidth * 72.0 / instdata->pixelsperinch;
            height = instdata->imageheight * 72.0 / instdata->pixelsperinch;
            instdata->surface = cairo_svg_surface_create(instdata->imagename, width, height);
            fmtname = "SVG";
            break;
        case CCFBIF_REC:
            /* Values will be given in (floating-point) points */
            extents.x = 0.0;
            extents.y = 0.0;
            extents.width = instdata->imagewidth * 72.0 / instdata->pixelsperinch;
            extents.height = height = instdata->imageheight * 72.0 / instdata->pixelsperinch;
#ifdef CAIRO_HAS_RECORDING_SURFACE
            instdata->surface = cairo_recording_surface_create(CAIRO_CONTENT_COLOR_ALPHA, &extents);
#else
            instdata->surface = cairo_svg_surface_create_for_stream(NULL, NULL, extents.width, extents.height);
#endif
            fmtname = "recording";
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
    }

    /* Create the Context if it does not exist */
    if ( instdata->context == NULL ) {
        instdata->context = cairo_create(instdata->surface);
        if ( cairo_status(instdata->context) != CAIRO_STATUS_SUCCESS ) {
            strcpy(grdelerrmsg, "cairoCFerBind_createSurface: "
                                "problems creating a context from a surface");
            cairo_destroy(instdata->context);
            instdata->context = NULL;
            cairo_surface_finish(instdata->surface);
            cairo_surface_destroy(instdata->surface);
            instdata->surface = NULL;
            return 0;
        }

        /*
         * If landscape PostScript, translate and rotate the coordinate system
         * to correct for swapped width and height (per Cairo requirements).
         */
        if ( instdata->imageformat == CCFBIF_PS ) {
            width = instdata->imagewidth * 72.0 / instdata->pixelsperinch;
            height = instdata->imageheight * 72.0 / instdata->pixelsperinch;
            if ( width > height ) {
                /* surface was created with coordinates (0,0) to (height, width) */
                cairo_matrix_t transmat;

                /* Add a "comment" telling PostScript it is landscape */
                cairo_ps_surface_dsc_begin_page_setup(instdata->surface);
                cairo_ps_surface_dsc_comment(instdata->surface,
                                         "%%PageOrientation: Landscape");
                /* Translate and rotate 90 degrees */
                cairo_matrix_init(&transmat, 0.0, -1.0, 1.0, 0.0, 0.0, width);
                cairo_set_matrix(instdata->context, &transmat);
                /*
                 * The transformed coordinate system goes from (0,0) at the top
                 * left corner to (width, height) at the bottom right corner.
                 */
            }
            else {
                /* Add a "comment" telling PostScript it is portrait */
                cairo_ps_surface_dsc_begin_page_setup(instdata->surface);
                cairo_ps_surface_dsc_comment(instdata->surface,
                                         "%%PageOrientation: Portrait");
            }
        }

        /* Set the antialiasing state in the context */
        if ( instdata->antialias )
            cairo_set_antialias(instdata->context, CAIRO_ANTIALIAS_DEFAULT);
        else
            cairo_set_antialias(instdata->context, CAIRO_ANTIALIAS_NONE);

        /* Set the appropriate clipping rectangle (if any) */
        if ( ! cairoCFerBind_clipView(self, instdata->clipit) ) {
            /* grdelerrmsg appropriately assigned */
            return 0;
        }
    }

    return 1;
}
