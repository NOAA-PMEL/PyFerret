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
 * Converts png filename string to null-terminated.
 * Stores converted string to CFerBindData struct member.
 * Arguments:
 *     filename  - name for the image file
 *     len_filename - actual length of filename (number of characters)
 *     xloc - horizontal coordinate of upper left image corner in final image
 *     yloc - vertical coordinate of upper left image corner in final image
 *     scalefrac - scaling of x and y dimensions of watermark image
 *     opacity - value in range [0.0, 1.0] corresponding to % visibility of watermark image
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */

grdelBool cairoCFerBind_setWaterMark(CFerBind *self, const char filename[], int len_filename,
                                     float xloc, float yloc, float scalefrac, float opacity)
{
    CairoCFerBindData *instdata;

    /* Sanity checks - this should NOT be called by the PyQtCairo engine */
    if ( self->enginename != CairoCFerBindName ) {
        strcpy(grdelerrmsg, "cairoCFerBind_setWaterMark: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    if ( len_filename > CCFB_NAME_SIZE ) {
      strcpy(grdelerrmsg, "cairoCFerBind_setWaterMark: filename exceeds maximum length");
      return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Convert filepath to a null-terminated string */
    strncpy(instdata->wmark_filename, filename, len_filename);
    instdata->wmark_filename[len_filename] = '\0';

    /* Set watermark display vals */
    instdata->xloc = xloc;
    instdata->yloc = yloc;
    instdata->scalefrac = scalefrac;
    instdata->opacity = opacity;

    return 1;
}
