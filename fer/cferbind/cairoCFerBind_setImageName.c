/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Assigns the name and format of the image file to be created.
 * This will determine the type of Cairo surface that will be
 * created.  Any existing surface and context will be deleted.
 *
 * Arguments:
 *     imagename  - name for the image file (can be NULL)
 *     imgnamelen - actual length of imagename (zero if NULL)
 *     formatname - name of the image format; currently supported
 *                  (case insensitive) values are "PNG", "PDF",
 *                  "PS", "EPS", "SVG", "", or NULL
 *     fmtnamelen - actual length of formatname (zero if NULL)
 *
 * If formatname is "" or NULL, the filename extension of imagename,
 * if it exists and is recognized, will determine the format.  If
 * the extension does not exist or is not recognized, a PNG surface
 * will be created.  A filename consisting of only an extension
 * (e.g., ".png") will be treated as not having an extension.
 *
 * If the PNG surface is created, the saveWindow function is used
 * to save the image.  Thus, imagename is only a default name that
 * may not be used.  For other surfaces, the saveWindow function
 * does nothing as the drawing is being written directly to file
 * and cannot be saved to another file.
 *
 * If Cairo 1.6 or later, encapsulated PostScript is always used;
 * otherwise, non-encapsulated PostScript is always used.
 * Thus, the PS and EPS formats are always the same.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_setImageName(CFerBind *self, const char imagename[],
                        int imgnamelen, const char formatname[], int fmtnamelen)
{
    int  j, k;
    char fmtext[8];
    CCFBImageFormat imageformat;
    CairoCFerBindData *instdata;

    /* Sanity checks */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_setImageName: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return 0;
    }
    if ( imgnamelen >= CCFB_NAME_SIZE ) {
        sprintf(grdelerrmsg, "cairoCFerBind_setImageName: "
                             "imgnamelen (%d) too large", imgnamelen);
        return 0;
    }

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

    /* Get the format type from the format name */
    if ( strcmp(fmtext, "PNG") == 0 ) {
        imageformat = CCFBIF_PNG;
    }
    else if ( strcmp(fmtext, "PDF") == 0 ) {
        imageformat = CCFBIF_PDF;
    }
    else if ( (strcmp(fmtext, "EPS") == 0) || (strcmp(fmtext, "ps") == 0) ) {
        imageformat = CCFBIF_EPS;
    }
    else if ( strcmp(fmtext, "SVG") == 0 ) {
        imageformat = CCFBIF_SVG;
    }
    else if ( fmtnamelen <= 0 ) {
        /* No format specified and unrecognized extension */
        imageformat = CCFBIF_PNG;
    }
    else {
        /* An unrecognized format was specified */
        sprintf(grdelerrmsg, "cairoCFerBind_setImageName: "
                             "unrecognized format '%s'", fmtext);
        return 0;
    }

    /* Update the instance data structure */
    instdata = (CairoCFerBindData *) self->instancedata;
    instdata->imageformat = imageformat;
    strncpy(instdata->imagename, imagename, imgnamelen);
    instdata->imagename[imgnamelen] = '\0';

    /* Delete any existing context and surface */
    if ( instdata->context != NULL ) {
        cairo_destroy(instdata->context);
        instdata->context = NULL;
    }
    if ( instdata->surface != NULL ) {
        cairo_surface_destroy(instdata->surface);
        instdata->surface = NULL;
    }
    instdata->somethingdrawn = 0;

    /*
     * Defer creating a new surface and context until beginView
     * is called.  Ferret may change size and other details before
     * actually starting to draw.
     */

    return 1;
}

