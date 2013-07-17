/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Ends a "View" for this "Window".
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_endView(CFerBind *self)
{
    CairoCFerBindData *instdata;
    CCFBPicture       *thispic;
    cairo_status_t     status;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_endView: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Nothing to do if neither an image nor recording surface */
    if ( (instdata->imageformat != CCFBIF_PNG) &&
         (instdata->imageformat != CCFBIF_REC) ) {
        return 1;
    }
     
    /* If something was drawn, delete the context but save the surface */
    if ( instdata->somethingdrawn ) {
        if ( instdata->context == NULL ) {
            strcpy(grdelerrmsg, "cairoCFerBind_endView: unexpected error, "
                                "something drawn without a context");
            return 0;
        }
        if ( instdata->surface == NULL ) {
            strcpy(grdelerrmsg, "cairoCFerBind_endView: unexpected error, "
                                "something drawn without a surface");
            return 0;
        }

        /* Allocate a new picture for the linked list */
        thispic = (CCFBPicture *) PyMem_Malloc(sizeof(CCFBPicture));
        if ( thispic == NULL ) {
            strcpy(grdelerrmsg, "cairoCFerBind_endView: "
                                "Out of memory for a CCFBPicture structure");
            return 0;
        }

        /* Only need the surface, not the context */
        /* cairo_show_page(instdata->context); - adds a newpage */
        /* Make sure the context is not in an error state */
        status = cairo_status(instdata->context);
        if ( status != CAIRO_STATUS_SUCCESS ) {
            sprintf(grdelerrmsg, "cairoCFerBind_endView: "
                                 "cairo context error: %s", 
                                 cairo_status_to_string(status));
            return 0;
        }
        cairo_destroy(instdata->context);
        instdata->context = NULL;

        cairo_surface_flush(instdata->surface);
        /* Make sure the surface is not in an error state */
        status = cairo_surface_status(instdata->surface);
        if ( status != CAIRO_STATUS_SUCCESS ) {
            sprintf(grdelerrmsg, "cairoCFerBind_endView: "
                                 "cairo surface error: %s", 
                                 cairo_status_to_string(status));
            return 0;
        }


        /* Assign the current surface and segment ID to this picture */
        thispic->next = NULL;
        thispic->surface = instdata->surface;
        thispic->segid = instdata->segid;

        instdata->surface = NULL;
        instdata->somethingdrawn = 0;

        /* Add the picture to the linked list */
        if ( instdata->lastpic == NULL ) {
            instdata->firstpic = thispic;
            instdata->lastpic = thispic;
        }
        else {
            instdata->lastpic->next = thispic;
            instdata->lastpic = thispic;
        }

    }

    return 1;
}

