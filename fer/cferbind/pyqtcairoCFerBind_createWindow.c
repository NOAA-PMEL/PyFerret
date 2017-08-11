/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "pyqtcairoCFerBind.h"
#include "FerMem.h"

/*
 * Creates a PipedImager (previously called PyQtCairo) 
 * instance of a CFerBind struct.
 *
 * A pointer to created bindings instance is returned if
 * successful.  If an error occurs, grdelerrmsg is assigned
 * an appropriate error message and NULL is returned.
 */
CFerBind *pyqtcairoCFerBind_createWindow(const char *windowname, int windnamelen,
                                         int visible, int noalpha)
{
    CFerBind *bindings;
    CairoCFerBindData *instdata;

    /* Create the bindings structure */
    bindings = (CFerBind *) FerMem_Malloc(sizeof(CFerBind), __FILE__, __LINE__);
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_createWindow: "
                            "Out of memory for a CFerBind structure");
        return NULL;
    }
    /* Zero out everything to catch errors */
    memset(bindings, 0, sizeof(CFerBind));

    /* Identify the type of bindings using the pointer address to the global name */
    bindings->enginename = PyQtCairoCFerBindName;

    /* binding functions specific for the PipedImager engine */
    bindings->setImageName = pyqtcairoCFerBind_setImageName;
    bindings->deleteWindow = pyqtcairoCFerBind_deleteWindow;
    bindings->updateWindow = pyqtcairoCFerBind_updateWindow;
    bindings->clearWindow = pyqtcairoCFerBind_clearWindow;
    bindings->redrawWindow = pyqtcairoCFerBind_redrawWindow;
    bindings->windowScreenInfo = pyqtcairoCFerBind_windowScreenInfo;
    bindings->setWindowDpi = NULL;
    bindings->resizeWindow = pyqtcairoCFerBind_resizeWindow;
    bindings->scaleWindow = pyqtcairoCFerBind_scaleWindow;
    bindings->showWindow = pyqtcairoCFerBind_showWindow;
    bindings->saveWindow = pyqtcairoCFerBind_saveWindow;

    /* binding functions shared with the Cairo engine */
    bindings->setAntialias = cairoCFerBind_setAntialias;
    bindings->beginView = cairoCFerBind_beginView;
    bindings->clipView = cairoCFerBind_clipView;
    bindings->endView = cairoCFerBind_endView;
    bindings->beginSegment = cairoCFerBind_beginSegment;
    bindings->endSegment = cairoCFerBind_endSegment;
    bindings->deleteSegment = cairoCFerBind_deleteSegment;
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
    bindings->setWidthFactor = cairoCFerBind_setWidthFactor;
    bindings->drawMultiline = cairoCFerBind_drawMultiline;
    bindings->drawPoints = cairoCFerBind_drawPoints;
    bindings->drawPolygon = cairoCFerBind_drawPolygon;
    bindings->drawRectangle = cairoCFerBind_drawRectangle;
    bindings->textSize = cairoCFerBind_textSize;
    bindings->drawText = cairoCFerBind_drawText;

    /* Create the instance-specific data structure */
    bindings->instancedata = 
        (CairoCFerBindData *) FerMem_Malloc(sizeof(CairoCFerBindData), __FILE__, __LINE__);
    if ( bindings->instancedata == NULL ) {
        strcpy(grdelerrmsg, "pyqtcairoCFerBind_createWindow: "
                            "Out of memory for a CairoCFerBindData structure");
        FerMem_Free(bindings, __FILE__, __LINE__);
        return NULL;
    }
    /* Initialize everything to zero */
    memset(bindings->instancedata, 0, sizeof(CairoCFerBindData));

    /* Set non-zero default values */
    instdata = (CairoCFerBindData *) bindings->instancedata;
    /* Cairo surface type - must be an image surface */
    instdata->imageformat = CCFBIF_PNG;
    /* default DPI, image size, line width scaling factor, and minimum allowed value */
    instdata->pixelsperinch = 96;
    instdata->imagewidth = (int) (10.2 * instdata->pixelsperinch);
    instdata->imageheight = (int) (8.8 * instdata->pixelsperinch);
    instdata->widthfactor = 0.72 * instdata->pixelsperinch / 72.0;
    instdata->minsize = 128;
    /* default clear color of opaque white */
    instdata->lastclearcolor.id = CCFBColorId;
    instdata->lastclearcolor.redfrac = 1.0;
    instdata->lastclearcolor.greenfrac = 1.0;
    instdata->lastclearcolor.bluefrac = 1.0;
    instdata->lastclearcolor.opaquefrac = 1.0;
    /* default line width scaling factor */

    /* save the decision about the alpha channel */
    instdata->noalpha = noalpha;

    /*
     * Get bindings to PipedImagerPQ for displaying the image.
     * This prevents duplication of Python-calling code for those
     * PipedImager methods interacting with the viewer.
     */
    instdata->viewer = grdelWindowCreate("PipedImagerPQ", 13, windowname,
                                         windnamelen, visible, noalpha, 1);
    if ( instdata->viewer == NULL ) {
        /* grdelerrmsg already assigned */
        FerMem_Free(bindings->instancedata, __FILE__, __LINE__);
        FerMem_Free(bindings, __FILE__, __LINE__);
        return NULL;
    }

    return bindings;
}

