/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
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
    cairo_status_t status;
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
    if ( (instdata->context == NULL) || (instdata->surface == NULL) ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: "
                            "attempting to update an empty image");
        return 0;
    }

    /* Make sure the context is not in an error state */
    status = cairo_status(instdata->context);
    if ( status != CAIRO_STATUS_SUCCESS ) {
        sprintf(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: "
                             "context has error state %d", status);
        return 0;
    }

    /* Make sure all drawing to the surface is completed (paranoia) */
    cairo_surface_flush(instdata->surface);

    /* Make sure the surface is not in an error state */
    status = cairo_surface_status(instdata->surface);
    if ( status != CAIRO_STATUS_SUCCESS ) {
        sprintf(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: "
                             "surface has error state %d", status);
        return 0;
    }

    /* Get the image dimension and data from the image surface */
    width = cairo_image_surface_get_width(instdata->surface);
    height = cairo_image_surface_get_height(instdata->surface);
    stride = cairo_image_surface_get_stride(instdata->surface);
    imagedata = cairo_image_surface_get_data(instdata->surface);
    if ( imagedata == NULL ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: "
                            "cairo_image_surface_get_data failed");
        return 0;
    }

    bindings = grdelWindowVerify(instdata->viewer);
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: unexpected error "
                            "viewer is not a grdelWindow");
        return 0;
    }

    /* Call the updateWindow method of the bindings instance. */
    databytearray = PyByteArray_FromStringAndSize((const char *) imagedata, stride * height);
    if ( databytearray == NULL ) {
        sprintf(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: error when creating "
                "the image data bytearray: %s", pyefcn_get_error());
        return 0;
    }
    result = PyObject_CallMethod(bindings->pyobject, "newSceneImage",
                                 "iiiN", width, height, stride, databytearray);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "pyqtcairoCFerBind_updateWindow: error when calling the "
                "Python binding's newSceneImage method: %s", pyefcn_get_error());
        return 0;
    }
    Py_DECREF(result);

    return 1;
}

