/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <cairo/cairo-pdf.h>
#include <cairo/cairo-ps.h>
#include <cairo/cairo-svg.h>
#include <pango/pangocairo.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "FerMem.h"

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
 *     annotations - array of annotation strings; pointers are always 8 bytes apart
 *     numannotations - number of annotation strings
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
 * If transbkg is zero, the saved image is filled with the
 * last clearing color before drawing the current image with a
 * transparent background.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */

grdelBool cairoCFerBind_saveWindow(CFerBind *self, const char *filename,
                        int namelen, const char *formatname, int fmtnamelen,
                        int transbkg, double xinches, double yinches,
                        int xpixels, int ypixels,
                        void **annotations, int numannotations)
{
    CairoCFerBindData *instdata;
    cairo_status_t     status;
    const char        *imagename;
    int                imgnamelen;
    int                j, k;
    char               fmtext[8];
    char              *allannos;
    cairo_surface_t   *annosurface;
    cairo_t           *annocontext;
    double             padding;
    double             penwidth;
    cairo_surface_t   *savesurface;
    cairo_t           *savecontext;
    PangoLayout       *annolayout;
    double             layoutwidth;
    double             layoutheight;
    cairo_status_t     result;
    char               savename[CCFB_NAME_SIZE];
    double             savewidth;
    double             saveheight;
    int                noalpha;
    double             scalefactor;
    double             offset;
    CCFBPicture       *thispic;
    cairo_rectangle_t  extents;

    /* Sanity checks - this should NOT be called by the PyQtCairo engine */
    if ( self->enginename != CairoCFerBindName ) {
        strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* This might be called with no images present; if so, ignore the call */
    if ( (instdata->surface == NULL) && (instdata->firstpic == NULL) ) {
        return 1;
    }

    /* Make sure the context is not in an error state */
    if ( instdata->context != NULL ) {
        status = cairo_status(instdata->context);
        if ( status != CAIRO_STATUS_SUCCESS ) {
            sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                                 "cairo context error: %s",
                                 cairo_status_to_string(status));
            return 0;
        }
    }

    /* Just to be safe */
    if ( instdata->surface != NULL ) {
        cairo_surface_flush(instdata->surface);

        /* Make sure the surface is not in an error state */
        status = cairo_surface_status(instdata->surface);
        if ( status != CAIRO_STATUS_SUCCESS ) {
            sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                                 "cairo surface error: %s",
                                 cairo_status_to_string(status));
            return 0;
        }
    }

    /*
     * Only allow annotations with recording surfaces.
     * Batch mode is deprecated and it is too complicated.
     */
    if ( (numannotations > 0) && (instdata->imageformat != CCFBIF_REC) ) {
        strcpy(grdelerrmsg, "Annotations cannot be used with batch mode");
        return 0;
    }

    /* Check the surface type -
     *    PNG from -png command line option
     *    PDF from metadata mode
     *    REC from normal opertaions
     */
    if ( (instdata->imageformat != CCFBIF_PNG) &&
         (instdata->imageformat != CCFBIF_PDF) &&
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

    if ( numannotations > 0 ) {
        /* Allocate memory for the string with all the annotations */
        for (k = 0, j = 0; k < numannotations; k++, j++)
            j += strlen((char *) annotations[k * 8 / sizeof(void *)]);
        allannos = (char *) FerMem_Malloc(j * sizeof(char), __FILE__, __LINE__);
        if ( allannos == NULL ) {
            strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: "
                   "out of memory for concatenated annotations");
            return 0;
        }
        /* Copy the annotations with newlines inbetween */
        for (k = 0, j = 0; k < numannotations; k++, j++) {
            strcpy(&(allannos[j]), (char *) annotations[k * 8 / sizeof(void *)]);
            j += strlen((char *) annotations[k * 8 / sizeof(void *)]);
            allannos[j] = '\n';
        }
        /* Remove the last newline */
        allannos[j-1] = '\0';
        /* padding and pen width in points */
        padding = 9.0;
        penwidth = 2.0;
        /*
         * Create the recording surface for the annotations;
         * keep the same width, less padding, as the image; the
         * height is actually arbitrary.
         */
        extents.x = 0.0;
        extents.y = 0.0;
        extents.width = instdata->imagewidth * 72.0 / instdata->pixelsperinch - 2.0 * padding;
        extents.height = instdata->imageheight * 72.0 / instdata->pixelsperinch - 2.0 * padding;
#if 0
#ifdef CAIRO_HAS_RECORDING_SURFACE
        annosurface = cairo_recording_surface_create(CAIRO_CONTENT_COLOR_ALPHA, &extents);
#else
        annosurface = cairo_svg_surface_create_for_stream(NULL, NULL, extents.width, extents.height);
#endif
#else
        /* Always use the SVG surface for the old Pango library */
        annosurface = cairo_svg_surface_create_for_stream(NULL, NULL, extents.width, extents.height);
#endif
        if ( cairo_surface_status(annosurface) != CAIRO_STATUS_SUCCESS ) {
            strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: "
                                "problems creating a temp surface for annotations");
            cairo_surface_destroy(annosurface);
            FerMem_Free(allannos, __FILE__, __LINE__);
            return 0;
        }
        annocontext = cairo_create(annosurface);
        if ( cairo_status(annocontext) != CAIRO_STATUS_SUCCESS ) {
            strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: "
                                "problems creating a temp context from a surface");
            cairo_destroy(annocontext);
            cairo_surface_finish(annosurface);
            cairo_surface_destroy(annosurface);
            FerMem_Free(allannos, __FILE__, __LINE__);
            return 0;
        }
        /* Create the Pango layout for the annotations */
        annolayout = pango_cairo_create_layout(annocontext);
        pango_layout_set_width(annolayout, (int) (PANGO_SCALE * (extents.width) + 0.5));
        pango_layout_set_wrap(annolayout, PANGO_WRAP_WORD_CHAR);
        pango_layout_set_markup(annolayout, allannos, j-1);
        /* Apply the annotations to this cairo surface */
        pango_cairo_show_layout(annocontext, annolayout);
        /* Get the actual size of the annotations in Pango scaled surface units */
        pango_layout_get_size(annolayout, NULL, &k);
        if ( k > 0 ) {
           layoutheight = (double) k / PANGO_SCALE + 2.0 * padding;
           layoutwidth = extents.width + 2.0 * padding;
        }
        else {
           /* No content in the annotations so ignore them */
           padding = 0.0;
           penwidth = 0.0;
           layoutheight = 0.0;
           layoutwidth = 0.0;
        }
        /* Done with the Pango layout */
        g_object_unref(annolayout);
        /* Make sure the context is not in an error state */
        status = cairo_status(annocontext);
        if ( status != CAIRO_STATUS_SUCCESS ) {
            sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                                 "cairo annotation context error: %s",
                                 cairo_status_to_string(status));
            cairo_destroy(annocontext);
            cairo_surface_finish(annosurface);
            cairo_surface_destroy(annosurface);
            FerMem_Free(allannos, __FILE__, __LINE__);
            return 0;
        }
        /* Only need the surface, not the context */
        cairo_destroy(annocontext);

        cairo_surface_flush(annosurface);
        /* Make sure the surface is not in an error state */
        status = cairo_surface_status(annosurface);
        if ( status != CAIRO_STATUS_SUCCESS ) {
            sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                                 "cairo annotation surface error: %s",
                                 cairo_status_to_string(status));
            cairo_surface_finish(annosurface);
            cairo_surface_destroy(annosurface);
            FerMem_Free(allannos, __FILE__, __LINE__);
            return 0;
        }
    }
    else {
        annosurface = NULL;
        padding = 0.0;
        penwidth = 0.0;
        layoutheight = 0.0;
        layoutwidth = 0.0;
    }

    /* Create a temporary surface for the desired format */
    if ( strcmp(fmtext, "PNG") == 0 ) {
        /* Surface size is given in integer pixels */
        savewidth = (double) xpixels;
        saveheight = (double) ypixels;
        scalefactor  = savewidth / instdata->imagewidth;
        scalefactor += saveheight / instdata->imageheight;
        if ( instdata->imageformat == CCFBIF_PNG ) {
            /* memory image is in pixels value given in imagewidth, imageheight */
            scalefactor *= 0.5;
        }
        else {
            /* recording surface is actually in points value of imagewidth, imageheight */
            scalefactor *= instdata->pixelsperinch / 144.0;
        }
        saveheight += scalefactor * layoutheight;
        if ( instdata->noalpha ) {
            savesurface = cairo_image_surface_create(CAIRO_FORMAT_RGB24,
                                      (int) savewidth, (int) saveheight);
            noalpha = 1;
        }
        else {
            savesurface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32,
                                      (int) savewidth, (int) saveheight);
            noalpha = 0;
        }
    }
    else if ( strcmp(fmtext, "PDF") == 0 ) {
        /* Surface size is given in (floating-point) points */
        savewidth = xinches * 72.0;
        saveheight = yinches * 72.0;
        scalefactor  = savewidth / instdata->imagewidth;
        scalefactor += saveheight / instdata->imageheight;
        /* recording surface is actually in points value of imagewidth, imageheight */
        scalefactor *= instdata->pixelsperinch / 144.0;
        saveheight += scalefactor * layoutheight;
        savesurface = cairo_pdf_surface_create(savename, savewidth, saveheight);
        /* Never use the alpha channel */
        noalpha = 1;
    }
    else if ( strcmp(fmtext, "EPS") == 0 ) {
        /* Surface size is given in (floating-point) points */
        savewidth = xinches * 72.0;
        saveheight = yinches * 72.0;
        scalefactor  = savewidth / instdata->imagewidth;
        scalefactor += saveheight / instdata->imageheight;
        /* recording surface is actually in points value of imagewidth, imageheight */
        scalefactor *= instdata->pixelsperinch / 144.0;
        saveheight += scalefactor * layoutheight;
        savesurface = cairo_ps_surface_create(savename, savewidth, saveheight);
        /* Never use the alpha channel */
        noalpha = 1;
    }
    else if ( strcmp(fmtext, "PS") == 0 ) {
        /* Surface size is given in (floating-point) points */
        savewidth = xinches * 72.0;
        saveheight = yinches * 72.0;
        scalefactor  = savewidth / instdata->imagewidth;
        scalefactor += saveheight / instdata->imageheight;
        /* recording surface is actually in points value of imagewidth, imageheight */
        scalefactor *= instdata->pixelsperinch / 144.0;
        saveheight += scalefactor * layoutheight;
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
        /* Never use the alpha channel */
        noalpha = 1;
    }
    else if ( strcmp(fmtext, "SVG") == 0 ) {
        /* Surface size is given in (floating-point) points */
        savewidth = xinches * 72.0;
        saveheight = yinches * 72.0;
        scalefactor  = savewidth / instdata->imagewidth;
        scalefactor += saveheight / instdata->imageheight;
        /* recording surface is actually in points value of imagewidth, imageheight */
        scalefactor *= instdata->pixelsperinch / 144.0;
        saveheight += scalefactor * layoutheight;
        savesurface = cairo_svg_surface_create(savename, savewidth, saveheight);
        noalpha = instdata->noalpha;
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
    else if ( strcmp(fmtext, "EPS") == 0 ) {
        cairo_ps_surface_set_eps(savesurface, 1);
    }

    /*
     * If not a transparent background, or if the alpha channel
     * is not supported, fill in the background (with an opaque
     * color if the alpha channel is not supported).
     */
    if ( (! transbkg) || noalpha ) {
        if ( noalpha )
            cairo_set_source_rgb(savecontext,
                                 instdata->lastclearcolor.redfrac,
                                 instdata->lastclearcolor.greenfrac,
                                 instdata->lastclearcolor.bluefrac);
        else
            cairo_set_source_rgba(savecontext,
                                  instdata->lastclearcolor.redfrac,
                                  instdata->lastclearcolor.greenfrac,
                                  instdata->lastclearcolor.bluefrac,
                                  instdata->lastclearcolor.opaquefrac);
        cairo_paint(savecontext);
    }

    /* Set the scale on the destination so the source will just fit. */
    cairo_scale(savecontext, scalefactor, scalefactor);

    /* Check if there are annotations (with content) to be drawn */
    if ( layoutheight > 0.0 ) {
        /*
         * Draw the annotations in a white-filled, black-outlined
         * rectangle at the top of the temporary surface.
         */
        cairo_new_path(savecontext);
        cairo_rectangle(savecontext, 0.5 * penwidth, 0.5 * penwidth,
                        layoutwidth - penwidth, layoutheight - penwidth);
        /* white fill */
        if ( noalpha )
            cairo_set_source_rgb(savecontext, 1.0, 1.0, 1.0);
        else
            cairo_set_source_rgba(savecontext, 1.0, 1.0, 1.0, 1.0);
        cairo_fill_preserve(savecontext);
        /* black outline */
        if ( noalpha )
            cairo_set_source_rgb(savecontext, 0.0, 0.0, 0.0);
        else
            cairo_set_source_rgba(savecontext, 0.0, 0.0, 0.0, 1.0);
        cairo_set_line_width(savecontext, penwidth);
        cairo_set_dash(savecontext, NULL, 0, 0.0);
        cairo_set_line_cap(savecontext, CAIRO_LINE_CAP_SQUARE);
        cairo_set_line_join(savecontext, CAIRO_LINE_JOIN_MITER);
        cairo_stroke(savecontext);
        /*
         * Draw the transparent-background annotations image
         * onto the save surface within the rectangle.
         */
        cairo_set_source_surface(savecontext, annosurface, padding, padding);
        cairo_paint(savecontext);
        cairo_surface_flush(savesurface);
        offset = layoutheight;
    }
    else {
        offset = 0.0;
    }

    if ( annosurface != NULL ) {
        /* Done with the annotation surface */
        cairo_surface_finish(annosurface);
        cairo_surface_destroy(annosurface);
    }

    /*
     * Draw the transparent-background images onto this
     * temporary surface, beneath any annotations rectangle.
     */
    for (thispic = instdata->firstpic; thispic != NULL; thispic = thispic->next) {
        cairo_set_source_surface(savecontext, thispic->surface, 0.0, offset);
        cairo_paint(savecontext);
    }
    if ( instdata->surface != NULL ) {
        cairo_set_source_surface(savecontext, instdata->surface, 0.0, offset);
        cairo_paint(savecontext);
    }

    /* If a watermark png file is given create the image, else do nothing */
    if ( instdata->wmark_filename[0] != '\0' ) {
        cairo_surface_t *wmark_surface;

        /* create watermark surface */
        wmark_surface = cairo_image_surface_create_from_png(instdata->wmark_filename);

        /* scale and position image */
        cairo_scale(savecontext, instdata->scalefrac, instdata->scalefrac);
        cairo_translate(savecontext, instdata->xloc, instdata->yloc);

        /* paint watermark with opacity fraction */
        cairo_set_source_surface(savecontext, wmark_surface, 0.0, 0.0);
        cairo_paint_with_alpha(savecontext, instdata->opacity);

        /* clear surface after painting */
        cairo_surface_destroy(wmark_surface);
    }

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
