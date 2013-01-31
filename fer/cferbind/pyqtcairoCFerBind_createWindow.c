/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"

/*
 * Creates a PyQtCario instance of a CFerBind struct.
 *
 * A pointer to created bindings instance is returned if
 * successful.  If an error occurs, grdelerrmsg is assigned
 * an appropriate error message and NULL is returned.
 */
CFerBind *pyqtcairoCFerBind_createWindow(const char *windowname, int windnamelen,
                                         int visible)
{
    CFerBind *bindings;
    CairoCFerBindData *instdata;

    /* Create the bindings structure */
    bindings = (CFerBind *) PyMem_Malloc(sizeof(CFerBind));
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_createWindow: "
                            "Out of memory for a CFerBind structure");
        return NULL;
    }
    /* Zero out everything to catch errors */
    memset(bindings, 0, sizeof(CFerBind));

    /* Identify the type of bindings using the pointer address to the global name */
    bindings->enginename = PyQtCairoCFerBindName;

    /* binding functions specific for the PyQtCairo engine */
    bindings->setImageName = pyqtcairoCFerBind_setImageName;
    bindings->deleteWindow = pyqtcairoCFerBind_deleteWindow;
    bindings->updateWindow = pyqtcairoCFerBind_updateWindow;
    bindings->clearWindow = pyqtcairoCFerBind_clearWindow;
    bindings->windowDpi = pyqtcairoCFerBind_windowDpi;
    bindings->resizeWindow = pyqtcairoCFerBind_resizeWindow;
    bindings->showWindow = pyqtcairoCFerBind_showWindow;
    bindings->saveWindow = pyqtcairoCFerBind_saveWindow;

    /* binding functions shared with the Cairo engine */
    bindings->setAntialias = cairoCFerBind_setAntialias;
    bindings->beginView = cairoCFerBind_beginView;
    bindings->clipView = cairoCFerBind_clipView;
    bindings->endView = cairoCFerBind_endView;
    bindings->createColor = cairoCFerBind_createColor;
    bindings->deleteColor = cairoCFerBind_deleteColor;
    bindings->createFont = cairoCFerBind_createFont;
    bindings->deleteFont = cairoCFerBind_deleteFont;
    bindings->createPen = cairoCFerBind_createPen;
    bindings->replacePenColor = cairoCFerBind_replacePenColor;
    bindings->deletePen = cairoCFerBind_deletePen;
    bindings->createBrush = cairoCFerBind_createBrush;
    bindings->replaceBrushColor = cairoCFerBind_replaceBrushColor;
    bindings->deleteBrush = cairoCFerBind_deleteBrush;
    bindings->createSymbol = cairoCFerBind_createSymbol;
    bindings->deleteSymbol = cairoCFerBind_deleteSymbol;
    bindings->drawMultiline = cairoCFerBind_drawMultiline;
    bindings->drawPoints = cairoCFerBind_drawPoints;
    bindings->drawPolygon = cairoCFerBind_drawPolygon;
    bindings->drawRectangle = cairoCFerBind_drawRectangle;
    bindings->drawText = cairoCFerBind_drawText;

    /* Create the instance-specific data structure */
    bindings->instancedata = 
        (CairoCFerBindData *) PyMem_Malloc(sizeof(CairoCFerBindData));
    if ( bindings->instancedata == NULL ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_createWindow: "
                            "Out of memory for a CairoCFerBindData structure");
        PyMem_Free(bindings);
        return NULL;
    }
    /* Initialize everything to zero */
    memset(bindings->instancedata, 0, sizeof(CairoCFerBindData));

    /* Set non-zero default values */
    instdata = (CairoCFerBindData *) bindings->instancedata;
    /* Cairo surface type - must be an image surface */
    instdata->imageformat = CCFBIF_PNG;
    /* image size and minimum allowed value */
    instdata->imagewidth = 840;
    instdata->imageheight = 720;
    instdata->minsize = 128;
    /* default clear color of opaque white */
    instdata->lastclearcolor.id = CCFBColorId;
    instdata->lastclearcolor.redfrac = 1.0;
    instdata->lastclearcolor.greenfrac = 1.0;
    instdata->lastclearcolor.bluefrac = 1.0;
    instdata->lastclearcolor.opaquefrac = 1.0;
    /* make sure the format is set correctly */
    instdata->imageformat = CCFBIF_PNG;
    /*
     * Get bindings to PyQtPipedImager for displaying the image.
     * This prevents duplication of Python-calling code for those
     * PyQtCairo methods interacting with the viewer.
     */
    instdata->viewer = grdelWindowCreate("PyQtImager", 10, windowname,
                                         windnamelen, visible);
    if ( instdata->viewer == NULL ) {
        /* grdelerrmsg already assigned */
        PyMem_Free(bindings->instancedata);
        PyMem_Free(bindings);
        return NULL;
    }

    return bindings;
}

