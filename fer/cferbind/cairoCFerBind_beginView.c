/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <cairo/cairo-pdf.h>
#include <cairo/cairo-ps.h>
#include <cairo/cairo-svg.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Starts a "View" for this "Window".
 * In this case (Cairo), this creates the surface and context,
 * if they do not leady exist, and assigns the clipping rectangle
 * that may be used for the subsequent drawings.
 *
 * Arguments:
 *     lftfrac - view left edge as a fraction [0.0, 1.0) of the window size
 *     btmfrac - view bottom edge as a fraction (0.0, 1.0] of the window size
 *     rgtfrac - view right edge as a fraction (0.0, 1.0] of the window size
 *     topfrac - view top edge as a fraction [0.0, 1.0) of the window size
 *     clipit  - value passed to cairoCFerBind_clipView
 *               (clip drawing to this view rectangle?)
 *
 * The fractions are from the top left corner; thus lftfrac must be less than
 * rgtfrac and topfrac must be less than btmfrac.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_beginView(CFerBind *self, double lftfrac, double btmfrac,
                                  double rgtfrac, double topfrac, int clipit)
{
    CairoCFerBindData *instdata;
    double width;
    double height;
    char  *fmtname;
    int    result;

    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_beginView: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Verify valid view fractions */
    if ( (0.0 > lftfrac) || (lftfrac >= rgtfrac) || (rgtfrac > 1.0) ){
        sprintf(grdelerrmsg, "cairoCFerBind_beginView: invalid left (%#.3f) "
                             "and/or right (%#.3f) fractions", lftfrac, rgtfrac);
        return 0;
    }
    if ( (0.0 > topfrac) || (topfrac >= btmfrac) || (btmfrac > 1.0) ){
        sprintf(grdelerrmsg, "cairoCFerBind_beginView: invalid top (%#.3f) "
                             "and/or bottom (%#.3f) fractions", topfrac, btmfrac);
        return 0;
    }

    /* Create the surface if it does not exist */
    if ( instdata->surface == NULL ) {
        /*
         * Sanity check: the context includes the surface, but cairoCFerBind
         * also maintains a separate reference to the surface; thus, if there
         * is no surface, there should not be a context.
         */
        if ( instdata->context != NULL ) {
            strcpy(grdelerrmsg, "cairoCFerBind_beginView: unexpected error, "
                                "NULL surface but non-NULL context");
            return 0;
        }
        /* Create the appropriate surface */
        switch( instdata->imageformat ) {
        case CCFBIF_PNG:
            /* Surface size is given in integer pixels */
            instdata->surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32,
                                      instdata->imagewidth, instdata->imageheight);
            /* Note that all surface values are initialized to zero (transparent) */
            fmtname = "PNG";
            break;
        case CCFBIF_PDF:
            /* Surface size is given in (floating-point) points */
            width = (double) instdata->imagewidth * CCFB_POINTS_PER_PIXEL;
            height = (double) instdata->imageheight * CCFB_POINTS_PER_PIXEL;
            instdata->surface = cairo_pdf_surface_create(instdata->imagename,
                                                         width, height);
            fmtname = "PDF";
            break;
        case CCFBIF_EPS:
            /* Surface size is given in (floating-point) points */
            width = (double) instdata->imagewidth * CCFB_POINTS_PER_PIXEL;
            height = (double) instdata->imageheight * CCFB_POINTS_PER_PIXEL;
            instdata->surface = cairo_ps_surface_create(instdata->imagename,
                                                        width, height);
            cairo_ps_surface_set_eps(instdata->surface, 1);
            fmtname = "EPS";
            break;
        case CCFBIF_SVG:
            /* Surface size is given in (floating-point) points */
            width = (double) instdata->imagewidth * CCFB_POINTS_PER_PIXEL;
            height = (double) instdata->imageheight * CCFB_POINTS_PER_PIXEL;
            instdata->surface = cairo_svg_surface_create(instdata->imagename,
                                                         width, height);
            fmtname = "SVG";
            break;
        default:
            sprintf(grdelerrmsg, "cairoCFerBind_beginView: unexpected error, "
                                 "unknown imageformat %d", instdata->imageformat);
            return 0;
        }
        /* Check for failure to create the surface */
        if ( cairo_surface_status(instdata->surface) != CAIRO_STATUS_SUCCESS ) {
            sprintf(grdelerrmsg, "cairoCFerBind_beginView: "
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
            strcpy(grdelerrmsg, "cairoCFerBind_beginView: "
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
    }

    /* Assign the view rectangle fractions */
    instdata->fracsides.left = lftfrac;
    instdata->fracsides.bottom = btmfrac;
    instdata->fracsides.right = rgtfrac;
    instdata->fracsides.top = topfrac;

    /* Assign the line width scaling factor for this view */
    width = rgtfrac - lftfrac;
    height = btmfrac - topfrac;
    instdata->viewfactor = sqrt(width * width + height * height);

    /* Assign clipping */
    result = self->clipView(self, clipit);

    return result;
}

