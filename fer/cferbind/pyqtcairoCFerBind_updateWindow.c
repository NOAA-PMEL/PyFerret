/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"
#include "pyferret.h"

/*
 * Sends the current cairo-generated image to the viewer to be displayed.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool pyqtcairoCFerBind_updateWindow(CFerBind *self)
{
    CairoCFerBindData *instdata;
    cairo_surface_t   *savesurface;
    cairo_t           *savecontext;
    cairo_status_t     status;
    CCFBPicture       *thispic;
    int width;
    int height;
    int stride;
    const unsigned char *imagedata;
    const BindObj *bindings;
    PyObject *result;
    PyObject *databytearray;

    /* Sanity check */
    if ( self->enginename != PyQtCairoCFerBindName ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    if ( ! instdata->imagechanged ) {
        /* Nothing new about the image; ignore the call */
        return 1;
    }
    if ( (instdata->surface == NULL) && (instdata->firstpic == NULL) ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: unexpected error, "
                            "trying to update an empty image");
        return 0;
    }

    /* Make sure the context is not in an error state */
    if ( instdata->context != NULL ) {
        status = cairo_status(instdata->context);
        if ( status != CAIRO_STATUS_SUCCESS ) {
            sprintf(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: "
                                 "cairo context error: %s", 
                                 cairo_status_to_string(status));
            return 0;
        }
    }

    if ( instdata->surface != NULL ) {
        /* Make sure all drawing to the surface is completed (paranoia) */
        cairo_surface_flush(instdata->surface);
        /* Make sure the surface is not in an error state */
        status = cairo_surface_status(instdata->surface);
        if ( status != CAIRO_STATUS_SUCCESS ) {
            sprintf(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: "
                                 "cairo surface error: %s", 
                                 cairo_status_to_string(status));
            return 0;
        }
    }

    if ( instdata->firstpic != NULL ) {
        /* create a temporary surface to combine all the pictures */
        savesurface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32,
                            instdata->imagewidth, instdata->imageheight);
        if ( cairo_surface_status(savesurface) != CAIRO_STATUS_SUCCESS ) {
            strcpy(grdelerrmsg, "pyqtCairoCFerBind_updateWindow: problems "
                                "creating a combined pictures image surface");
            cairo_surface_finish(savesurface);
            cairo_surface_destroy(savesurface);
            return 0;
        }
        savecontext = cairo_create(savesurface);
        if ( cairo_status(savecontext) != CAIRO_STATUS_SUCCESS ) {
            strcpy(grdelerrmsg, "pyqtCairoCFerBind_updateWindow: problems creating "
                                "a context for the combined pictures image surface");
            cairo_destroy(savecontext);
            cairo_surface_finish(savesurface);
            cairo_surface_destroy(savesurface);
            return 0;
        }
        /* Draw the transparent-background images onto this temporary surface */
        for (thispic = instdata->firstpic; thispic != NULL; thispic = thispic->next) {
            cairo_set_source_surface(savecontext, thispic->surface, 0.0, 0.0);
            cairo_paint(savecontext);
        }
        if ( instdata->surface != NULL ) {
            cairo_set_source_surface(savecontext, instdata->surface, 0.0, 0.0);
            cairo_paint(savecontext);
        }
        /* Just to be safe */
        cairo_show_page(savecontext);
        /* Done with the temporary context */
        cairo_destroy(savecontext);
        /* Just to be safe */
        cairo_surface_flush(savesurface);
    }
    else {
        /* Just use the one current surface */
        savesurface = instdata->surface;
    }

    /* Get the image dimension and data from the image surface */
    width = cairo_image_surface_get_width(savesurface);
    height = cairo_image_surface_get_height(savesurface);
    stride = cairo_image_surface_get_stride(savesurface);
    imagedata = cairo_image_surface_get_data(savesurface);
    if ( imagedata == NULL ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: "
                            "cairo_image_surface_get_data failed");
        if ( savesurface != instdata->surface ) {
            cairo_surface_finish(savesurface);
            cairo_surface_destroy(savesurface);
        }
        return 0;
    }

    bindings = grdelWindowVerify(instdata->viewer);
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: unexpected error "
                            "viewer is not a grdelWindow");
        if ( savesurface != instdata->surface ) {
            cairo_surface_finish(savesurface);
            cairo_surface_destroy(savesurface);
        }
        return 0;
    }

    /* Call the updateWindow method of the bindings instance. */
    databytearray = PyByteArray_FromStringAndSize((const char *) imagedata, stride * height);
    if ( databytearray == NULL ) {
        sprintf(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: error when creating "
                "the image data bytearray: %s", pyefcn_get_error());
        if ( savesurface != instdata->surface ) {
            cairo_surface_finish(savesurface);
            cairo_surface_destroy(savesurface);
        }
        return 0;
    }
    result = PyObject_CallMethod(bindings->pyobject, "newSceneImage",
                                 "iiiN", width, height, stride, databytearray);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: error when calling the "
                "Python binding's newSceneImage method: %s", pyefcn_get_error());
        if ( savesurface != instdata->surface ) {
            cairo_surface_finish(savesurface);
            cairo_surface_destroy(savesurface);
        }
        return 0;
    }
    Py_DECREF(result);

    if ( savesurface != instdata->surface ) {
        cairo_surface_finish(savesurface);
        cairo_surface_destroy(savesurface);
    }

    instdata->imagechanged = 0;

    return 1;
}

