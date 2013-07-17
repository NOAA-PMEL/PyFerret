/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Starts a "Segment" for this "Window".
 * A "Segment" is a group of drawing commands.
 *
 * Arguments:
 *     segid  - ID for this segment.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_beginSegment(CFerBind *self, int segid)
{
    CairoCFerBindData *instdata;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_beginSegment: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Ignore this call if not an image or recording surface */
    if ( (instdata->imageformat != CCFBIF_PNG) &&
         (instdata->imageformat != CCFBIF_REC) ) {
        return 1;
    }
     
    /* If something drawn, create that picture with the old segment ID */
    if ( instdata->somethingdrawn ) {
        if ( ! cairoCFerBind_endView(self) ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }

    /* assign the new segment ID to use */
    instdata->segid = segid;

    return 1;
}

