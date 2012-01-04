/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Saves this "Window" to file.
 * In this case (Cairo), this function currently only saves image
 * surfaces to PNG files.  All other surfaces return an error since
 * they write directly to the image file assigned when they were
 * created.
 *
 * Arguments:
 *     filename   - name of the image file to create, or an
 *                  empty string or NULL
 *     namelen    - actual length of filename (zero if NULL)
 *     formatname - name of the image format (case insensitive);
 *                  currently only "PNG", "", and NULL are supported.
 *     fmtnamelen - actual length of format (zero if NULL)
 *     transbkg   - leave the background transparent?
 *
 * If filename is empty or NULL, the imagename argument for the
 * last call to cairoCFerBind_setImageName is used for the
 * filename.
 *
 * If format is empty or NULL, the image format is determined
 * from the extension, of the filename.  In this case it is
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
grdelBool cairoCFerBind_saveWindow(CFerBind *self, const char *filename, int namelen,
                                   const char *formatname, int fmtnamelen, int transbkg)
{
    CairoCFerBindData *instdata;
    const char        *imagename;
    int                imgnamelen;
    int                j, k;
    char               fmtext[8];
    cairo_surface_t   *savesurface;
    cairo_status_t     result;
    char               savename[CCFB_NAME_SIZE];

    /* Sanity checks */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    if ( (instdata->surface == NULL) || (instdata->context == NULL) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: "
                            "attempting to save an empty image");
        return 0;
    }

    /* Check the surface type */
    if ( instdata->imageformat != CCFBIF_PNG ) {
        strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: surface is not a PNG image");
        return 0;
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

    /* Currently only PNG format is supported */
    if ( strcmp(fmtext, "PNG") != 0 ) {
        sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                             "unrecognized format '%s'", fmtext);
        return 0;
    }

    if ( ! transbkg ) {
        cairo_t *tempcontext;

        /* Create a temporary image surface */
        savesurface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32,
                            instdata->imagewidth, instdata->imageheight);
        if ( cairo_surface_status(savesurface) != CAIRO_STATUS_SUCCESS ) {
            strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: "
                                "problems creating a temporary PNG surface");
            cairo_surface_destroy(savesurface);
            return 0;
        }

        /* Create a temporary context for this tempoary surface */
        tempcontext = cairo_create(savesurface);
        if ( cairo_status(tempcontext) != CAIRO_STATUS_SUCCESS ) {
            strcpy(grdelerrmsg, "cairoCFerBind_saveWindow: problems creating "
                                "a tempoary context for the temporary surface");
            cairo_destroy(tempcontext);
            cairo_surface_destroy(savesurface);
            return 0;
        }

        /* Fill with the last clearing color */
        cairo_set_source_rgba(tempcontext,
                              instdata->lastclearcolor.redfrac,
                              instdata->lastclearcolor.greenfrac,
                              instdata->lastclearcolor.bluefrac,
                              instdata->lastclearcolor.opaquefrac);
        cairo_paint(tempcontext);

        /* Create a path covering the entire image */
        cairo_new_path(tempcontext);
        cairo_rectangle(tempcontext, 0.0, 0.0, 
                        (double) instdata->imagewidth,
                        (double) instdata->imageheight);

        /* Draw the transparent-background image onto this temporary surface */
        cairo_set_source_surface(tempcontext, instdata->surface, 0.0, 0.0);
        cairo_fill(tempcontext);

        /* No longer need the temporary context */
        cairo_destroy(tempcontext);
    }
    else
        /* Just use the transparent-background image as-is */
        savesurface = instdata->surface;

    /* Save the image to file */
    result = cairo_surface_write_to_png(savesurface, savename);
    switch( result ) {
    case CAIRO_STATUS_SUCCESS:
        break;
    case CAIRO_STATUS_WRITE_ERROR:
        sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                             "I/O error while saving to '%s'", savename);
        return 0;
    case CAIRO_STATUS_NO_MEMORY:
        sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: "
                             "out of memory while saving to '%s'", savename);
        return 0;
    case CAIRO_STATUS_SURFACE_TYPE_MISMATCH:
        sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: unexpected error, "
                             "type mismatch while saving to '%s'", savename);
        return 0;
    default:
        sprintf(grdelerrmsg, "cairoCFerBind_saveWindow: unexpected error, "
                             "unknown error %d while saving to '%s'",
                             result, savename);
        return 0;
    }

    /* Delete the temporary surface if it was created */
    if ( savesurface != instdata->surface )
        cairo_surface_destroy(savesurface);
    return 1;
}

