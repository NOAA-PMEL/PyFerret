/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"

/*
 * Assigns the default name and format for saving the image
 * from the PipedImagerPQ window.
 *
 * Arguments:
 *     imagename  - name for the image file (can be NULL)
 *     imgnamelen - actual length of imagename (zero if NULL)
 *     formatname - name of the image format (case insensitive);
 *                  (can be NULL)
 *     fmtnamelen - actual length of formatname (zero if NULL)
 *
 * If formatname is "" or NULL, the filename extension of
 * imagename, if it exists and is recognized, will determine
 * the format.  Only raster formats are supported.
 *
 * The saveWindow function is used to save the (possibly
 * scaled) image from PipedImagerPQ.  Thus, imagename
 * and formatname are only defaults that may not be used.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool pyqtcairoCFerBind_setImageName(CFerBind *self, const char imagename[],
                         int imgnamelen, const char formatname[], int fmtnamelen)
{
    CairoCFerBindData *instdata;
    grdelBool success;

    /* Sanity checks */
    if ( self->enginename != PyQtCairoCFerBindName ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_setImageName: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Send the information to the image display engine */
    success = grdelWindowSetImageName(instdata->viewer, imagename,
                                      imgnamelen, formatname, fmtnamelen);
    if ( ! success ) {
        /* grdelerrmsg already assigned */
        return 0;
    }
    /*
     * A cairo_image_surface (CCFBIF_PNG) is always created and
     * the image data is sent to PipedImagerPQ for display
     * and saving to file.  Thus the information does not need
     * to be recorded in the cairoCFerBindData structure.
     *
     * Since an image surface is always used, this call by itself
     * does not trigger the need for a new surface or context if
     * they exist.  If they do not exist, defer creating a new
     * surface and context until a drawing request is made.
     * Ferret may change size and other details before actually
     * starting to draw.
     */
    return 1;
}

