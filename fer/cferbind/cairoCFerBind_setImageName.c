/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "FerMem.h"

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
 *                  "PS", "SVG", "", or NULL
 *     fmtnamelen - actual length of formatname (zero if NULL)
 *
 * If formatname is "" or NULL, the filename extension of imagename,
 * if it exists and is recognized, will determine the format.  If
 * the extension does not exist or is not recognized, a recording
 * PostScript surface will be created.  A filename consisting of 
 * only an extension (e.g., ".png") will be treated as not having 
 * an extension.
 *
 * A "GIF" format is silently converted to "PNG".  A "PLT" format
 * is silently converted to "PDF".
 *
 * If a recording or an image surface is created, the saveWindow 
 * function is used to save the image.  Thus, imagename is only 
 * a default name that may not be used.  For other surfaces, the 
 * saveWindow function does nothing as the drawing is being written 
 * directly to file and cannot be saved to another file.
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
    CCFBPicture *delpic;

    /* Sanity checks - this should NOT be called by the PyQtCairo engine */
    if ( self->enginename != CairoCFerBindName ) {
        strcpy(grdelerrmsg, "cairoCFerBind_setImageName: unexpected error, "
                            "self is not a valid CFerBind struct");
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
    if ( (strcmp(fmtext, "PNG") == 0) || (strcmp(fmtext, "GIF") == 0) ) {
        imageformat = CCFBIF_PNG;
    }
    else if ( (strcmp(fmtext, "PDF") == 0) || (strcmp(fmtext, "PLT") == 0) ) {
        imageformat = CCFBIF_PDF;
    }
    else if ( strcmp(fmtext, "PS") == 0 ) {
        imageformat = CCFBIF_PS;
    }
    else if ( strcmp(fmtext, "SVG") == 0 ) {
        imageformat = CCFBIF_SVG;
    }
    else if ( fmtnamelen <= 0 ) {
        /* No format specified and unrecognized extension */
        imageformat = CCFBIF_REC;
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

    if ( strcmp(fmtext, "GIF") == 0 ) {
        /* Change .gif filename extension to .png */
        if ( (imgnamelen >= 4) &&
             (strcasecmp(&(instdata->imagename[imgnamelen-4]), ".gif") == 0) )
            strcpy(&(instdata->imagename[imgnamelen-4]), ".png");
    }
    else if ( strcmp(fmtext, "PLT") == 0 ) {
        /* Change .plt filename extension to .pdf */
        if ( (imgnamelen >= 4) &&
             (strcasecmp(&(instdata->imagename[imgnamelen-4]), ".plt") == 0) )
            strcpy(&(instdata->imagename[imgnamelen-4]), ".pdf");
    }

    /* Delete any existing context and surface */
    if ( instdata->context != NULL ) {
        cairo_destroy(instdata->context);
        instdata->context = NULL;
    }
    if ( instdata->surface != NULL ) {
        cairo_surface_finish(instdata->surface);
        cairo_surface_destroy(instdata->surface);
        instdata->surface = NULL;
    }
    instdata->somethingdrawn = 0;

    /* Delete any existing linked-list pictures */
    while ( instdata->firstpic != NULL ) {
        delpic = instdata->firstpic;
        instdata->firstpic = delpic->next;
        cairo_surface_finish(delpic->surface);
        cairo_surface_destroy(delpic->surface);
        FerMem_Free(delpic, __FILE__, __LINE__);
    }
    instdata->lastpic = NULL;
    /*
     * Defer creating a new surface and context until a drawing
     * request is made.  Ferret may change size and other details
     * before actually starting to draw.
     */

    return 1;
}

