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
 * Saves this "Window" to file.
 *
 * Arguments:
 *     filename   - name of the image file to create, 
 *                  or an empty string or NULL
 *     namelen    - actual length of filename (zero if NULL)
 *     formatname - name of the image format (case insensitive)
 *     fmtnamelen - actual length of format (zero if NULL)
 *     transbkg   - leave the background transparent?
 *     xinches    - horizontal size of vector image in inches
 *     yinches    - vertical size of vector image in inches
 *     xpixels    - horizontal size of raster image in pixels
 *     ypixels    - vertical size of raster image in pixels
 *
 * If filename is empty or NULL, the imagename argument for the
 * last call to cairoCFerBind_setImageName is used for the
 * filename.
 *
 * If format is empty or NULL, the image format is determined
 * from the extension of the filename.  In this case it is
 * an error if the extension does not exist or is not recognized.
 * A filename consisting of only an extension (e.g., ".png")
 * will be treated as not having an extension.
 *
 * If transbkg is non-zero, a temporary image is created, filled
 * with the last clearing color, and then the current image (with
 * a transparent background) is drawn onto this temporary image.
 * The temporary image is then saved to file.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_saveWindow(CFerBind *self, const char *filename, 
                        int namelen, const char *formatname, int fmtnamelen, 
                        int transbkg, double xinches, double yinches, 
                        int xpixels, int ypixels)
{
    CairoCFerBindData *instdata;
    const char        *imagename;
    int                imgnamelen;
    int                j, k;
    char               fmtext[8];
    cairo_surface_t   *savesurface;
    cairo_t           *savecontext;
    cairo_status_t     result;
    char               savename[CCFB_NAME_SIZE];
    double             savewidth;
    double             saveheight;
    int                usealpha;

    /* Sanity checks - this should NOT be called by the PyQtCairo engine */
    if ( self->enginename != CairoCFerBindName ) {
        strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    /* This might be called with no image present; if so, ignore the call */
    if ( (instdata->surface == NULL) || (instdata->context == NULL) ) {
        return 1;
    }
    /* Just to be safe */
    cairo_surface_flush(instdata->surface);

    /* Check the surface type */
    if ( (instdata->imageformat != CCFBIF_PNG) &&
         (instdata->imageformat != CCFBIF_REC) ) {
        /* Silently ignore this command since this was probably called automatically */
        return 1;
    }

    /* Get the image filename to use */
    if ( namelen > 0 ) {
        imagename = filename;
        imgnamelen = namelen;
    }
    else {
        imagename = instdata->imagename;
        imgnamelen = strlen(instdata->imagename);
    }

    /* Check that a filename was given somewhere */
    if ( imgnamelen <= 0 ) {
        strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: "
                            "unable to obtain a name for the image file");
        return 0;
    }
    if ( imgnamelen >= CCFB_NAME_SIZE ) {
        strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: "
                            "filename too long for this program");
        return 0;
    }
    /* Make a null-terminated copy of the filename */
    strncpy(savename, imagename, imgnamelen);
    savename[imgnamelen] = '\0';

    /* Get the format name */
    if ( fmtnamelen > 0 ) {
        /* lowercase, null-terminated, and should be short */
        for (j = 0; (j < 7) && (j < fmtnamelen); j++)
            fmtext[j] = (char) toupper(formatname[j]);
        fmtext[j] = '\0';
    }
    else {
        /*
         * Determine the format name based on the filename extension.
         * Ignore filenames that only have a '.' at the beginning
         * since these have been used for "not given" in Ferret.
         */
        for (k = imgnamelen - 1; k > 0; k--)
            if ( imagename[k] == '.' )
                break;
        if ( k > 0 ) {
            for (j = 0, k++; (j < 7) && (k < imgnamelen); j++, k++)
                fmtext[j] = (char) toupper(imagename[k]);
            fmtext[j] = '\0';
        }
        else
            fmtext[0] = '\0';
    }

    if ( strcmp(fmtext, "GIF") == 0 ) {
        strcpy(fmtext, "PNG");
        /* Change  .gif filename extension to .png */
        if ( (imgnamelen >= 4) &&
             (strcasecmp(&(savename[imgnamelen-4]), ".gif") == 0) )
            strcpy(&(savename[imgnamelen-4]), ".png");
    }
    else if ( strcmp(fmtext, "PLT") == 0 ) {
        strcpy(fmtext, "PDF");
        /* Change .plt filename extension to .pdf */
        if ( (imgnamelen >= 4) &&
             (strcasecmp(&(savename[imgnamelen-4]), ".plt") == 0) )
            strcpy(&(savename[imgnamelen-4]), ".pdf");
    }

    /* If an image surface, can only output PNG */
    if ( (instdata->imageformat == CCFBIF_PNG) &&
         (strcmp(fmtext, "PNG") != 0) ) {
        sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                "unrecognized format '%s' for an image surface", fmtext);
        return 0;
    }

    /* Create a temporary surface for the desired format */
    if ( strcmp(fmtext, "PNG") == 0 ) {
        /* Surface size is given in integer pixels */
        savesurface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32,
                                                 xpixels, ypixels);
        savewidth = (double) xpixels;
        saveheight = (double) ypixels;
        usealpha = 1;
    }
    else if ( strcmp(fmtext, "PDF") == 0 ) {
        /* Surface size is given in (floating-point) points */
        savewidth = xinches * 72.0;
        saveheight = yinches * 72.0;
        savesurface = cairo_pdf_surface_create(savename, savewidth, saveheight);
        usealpha = 0;
    }
    else if ( strcmp(fmtext, "PS") == 0 ) {
        /* Surface size is given in (floating-point) points */
        savewidth = xinches * 72.0;
        saveheight = yinches * 72.0;
        if ( savewidth > saveheight ) {
            /*
             * Landscape orientation
             * Swap savewidth and saveheight and then translate and rotate 
             * (see below) per Cairo requirements.
             */
            savesurface = cairo_ps_surface_create(savename, saveheight, savewidth);
        }
        else {
            /* Portrait orientation */
            savesurface = cairo_ps_surface_create(savename, savewidth, saveheight);
        }
        /* Do not use alpha channel - prevents embedded image */
        usealpha = 0;
    }
    else if ( strcmp(fmtext, "SVG") == 0 ) {
        /* Surface size is given in (floating-point) points */
        savewidth = xinches * 72.0;
        saveheight = yinches * 72.0;
        savesurface = cairo_svg_surface_create(savename, savewidth, saveheight);
        usealpha = 1;
    }
    else {
        sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                             "unrecognized format '%s'", fmtext);
        return 0;
    }

    /* Check for failure to create the surface */
    if ( cairo_surface_status(savesurface) != CAIRO_STATUS_SUCCESS ) {
        sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                             "problems creating a temporary %s surface", fmtext);
        cairo_surface_destroy(savesurface);
        return 0;
    }

    /* set the resolution for fallback raster images in vector drawings */
    cairo_surface_set_fallback_resolution(savesurface,
                      (double) CCFB_WINDOW_DPI, (double) CCFB_WINDOW_DPI);

    /* Create a temporary context for this temporary surface */
    savecontext = cairo_create(savesurface);
    if ( cairo_status(savecontext) != CAIRO_STATUS_SUCCESS ) {
        strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: problems creating "
                            "a temporary context for the temporary surface");
        cairo_destroy(savecontext);
        cairo_surface_destroy(savesurface);
        return 0;
    }

    /* 
     * Set the scale on the destination so the source will just fit. 
     * Note that imagewidth and imageheight are always in units of pixels.
     */
    if ( strcmp(fmtext, "PNG") == 0 ) {
        if ( instdata->imageformat == CCFBIF_PNG ) {
            /* 
             * savewidth, saveheight, imagewidth, and imageheight
             * are all in units of pixels.  Both surface use units
             * of pixels, so just scale to make the image fit.
             */
            cairo_scale(savecontext, 
                        savewidth / instdata->imagewidth,
                        saveheight / instdata->imageheight);
        }
        else {
            /*
             * savewidth, saveheight, imagewidth, and imageheight
             * are all in units of pixels.  However, the recording 
             * surface used units of points but this PNG surface 
             * uses units of pixels, so include that factor in the 
             * scaling.
             */
            cairo_scale(savecontext, 
                        savewidth / (instdata->imagewidth * CCFB_POINTS_PER_PIXEL), 
                        saveheight / (instdata->imageheight * CCFB_POINTS_PER_PIXEL));
        }
    }
    else {
        /*
         * savewidth and saveheight are in units of points, but
         * imagewidth and imageheight are in units of pixels.
         * Both surfaces use points so just need to convert 
         * imagewidth and imageheight to points for the scaling 
         * factor.
         */
        cairo_scale(savecontext, 
                    savewidth / (instdata->imagewidth * CCFB_POINTS_PER_PIXEL), 
                    saveheight / (instdata->imageheight * CCFB_POINTS_PER_PIXEL));
    }

    /*
     * If landscape PostScript, translate and rotate the coordinate system
     * to correct for swapped savewidth and saveheight (per Cairo requirements).
     */
    if ( strcmp(fmtext, "PS") == 0 ) {
       if ( savewidth > saveheight ) {
            /* surface was created with coordinates (0,0) to (saveheight, savewidth) */
            cairo_matrix_t transmat;

            /* Add a "comment" telling PostScript it is landscape */
            cairo_ps_surface_dsc_begin_page_setup(savesurface);
            cairo_ps_surface_dsc_comment(savesurface,
                                         "%%PageOrientation: Landscape");
            /* Translate and rotate 90 degrees */
            cairo_matrix_init(&transmat, 0.0, -1.0, 1.0, 0.0, 0.0, savewidth);
            cairo_set_matrix(savecontext, &transmat);
            /*
             * The transformed coordinate system goes from (0,0) at the top
             * left corner to (savewidth, saveheight) at the bottom right corner.
             */
        }
        else {
            /* Add a "comment" telling PostScript it is portrait */
            cairo_ps_surface_dsc_begin_page_setup(savesurface);
            cairo_ps_surface_dsc_comment(savesurface,
                                         "%%PageOrientation: Portrait");
        }
    }

    if ( ! transbkg ) {
        /* Fill with the last clearing color */
        if ( usealpha )
            cairo_set_source_rgba(savecontext,
                                  instdata->lastclearcolor.redfrac,
                                  instdata->lastclearcolor.greenfrac,
                                  instdata->lastclearcolor.bluefrac,
                                  instdata->lastclearcolor.opaquefrac);
        else
            cairo_set_source_rgb(savecontext,
                                 instdata->lastclearcolor.redfrac,
                                 instdata->lastclearcolor.greenfrac,
                                 instdata->lastclearcolor.bluefrac);
        cairo_paint(savecontext);
    }

    /* Draw the transparent-background image onto this temporary surface */
    cairo_set_source_surface(savecontext, instdata->surface, 0.0, 0.0);
    cairo_paint(savecontext);

    /* Just to be safe */
    cairo_show_page(savecontext);

    /* Done with the temporary context */
    cairo_destroy(savecontext);

    /* Just to be safe */
    cairo_surface_flush(savesurface);

    if ( strcmp(fmtext, "PNG") == 0 ) {
        /* Save the raster image in memory to file */
        result = cairo_surface_write_to_png(savesurface, savename);
        /* Done with the temporary surface */
        cairo_surface_finish(savesurface);
        cairo_surface_destroy(savesurface);
    }
    else {
        /* 
         * Vector images are written directly to file.
         * Check there were no errors after finishing 
         * off the surface.
         */
        cairo_surface_finish(savesurface);
        result = cairo_surface_status(savesurface);
        /* Done with the temporary surface */
        cairo_surface_destroy(savesurface);
    }

    /* Analyze the results */
    if ( result != CAIRO_STATUS_SUCCESS ) {
        sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                             "error while saving to '%s'; %s",
                             savename, cairo_status_to_string(result));
        return 0;
    }

    return 1;
}

