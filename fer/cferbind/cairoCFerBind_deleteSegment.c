/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"
#include "FerMem.h"

/*
 * Deletes the drawing commands in the indicated Segment of a Window. 
 *
 * Arguments:
 *     segid - ID of the segment to be deleted
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_deleteSegment(CFerBind *self, int segid)
{
    CairoCFerBindData *instdata;
    CCFBPicture *thispic;
    CCFBPicture *delpic;
    grdelBool    success;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deleteSegment: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Raise an error if not an image or recording surface */
    if ( (instdata->imageformat != CCFBIF_PNG) &&
         (instdata->imageformat != CCFBIF_REC) ) {
        strcpy(grdelerrmsg, "Unable to delete drawing segments when "
                            "writing directly to an image file");
        return 0;
    }
     
    /*
     * End the current picture if it has something drawn 
     * and is in the segment to delete 
     */
    if ( instdata->somethingdrawn && (instdata->segid == segid) ) {
        if ( ! cairoCFerBind_endView(self) ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }

    while ( (instdata->firstpic != NULL) && 
            (instdata->firstpic->segid == segid) ) {
        delpic = instdata->firstpic;
        instdata->firstpic = delpic->next;
        cairo_surface_finish(delpic->surface);
        cairo_surface_destroy(delpic->surface);
        FerMem_Free(delpic, __FILE__, __LINE__);
        instdata->imagechanged = 1;
    }
    instdata->lastpic = NULL;
    thispic = instdata->firstpic;
    while ( thispic != NULL )  {
        instdata->lastpic = thispic;
        if ( (thispic->next != NULL) &&
             (thispic->next->segid == segid) ) {
            delpic = thispic->next;
            thispic->next = delpic->next;
            cairo_surface_finish(delpic->surface);
            cairo_surface_destroy(delpic->surface);
            FerMem_Free(delpic, __FILE__, __LINE__);
            instdata->imagechanged = 1;
        }
        else {
            thispic = thispic->next;
        }
    }

    /* 
     * If PyQtCairo and a change was made, 
     * the image displayed needs to be updated.
     */
    if ( instdata->imagechanged && 
         (self->enginename == PyQtCairoCFerBindName) ) {
        success = pyqtcairoCFerBind_updateWindow(self);
        if ( ! success ) {
            /* grdelerrmsg aleady assigned */
            return 0;
        }
    }

    return 1;
}

