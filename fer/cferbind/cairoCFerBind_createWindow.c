/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Creates a Cario instance of a CFerBind struct.
 *
 * A pointer to created bindings instance is returned if
 * successful.  If an error occurs, grdelerrmsg is assigned
 * an appropriate error message and NULL is returned.
 */
CFerBind *cairoCFerBind_createWindow(void)
{
    CFerBind *bindings;
    CairoCFerBindData *instdata;

    /* Create the bindings structure */
    bindings = (CFerBind *) PyMem_Malloc(sizeof(CFerBind));
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createWindow: "
                            "Out of memory for a CFerBind structure");
        return NULL;
    }
    /* Zero out everything to catch errors */
    memset(bindings, 0, sizeof(CFerBind));

    /* Identify the type of bindings using the pointer address to the global name */
    bindings->enginename = CairoCFerBindName;

    /* Assign the bindings functions */
    bindings->setImageName = cairoCFerBind_setImageName;
    bindings->deleteWindow = cairoCFerBind_deleteWindow;
    bindings->setAntialias = cairoCFerBind_setAntialias;
    bindings->beginView = cairoCFerBind_beginView;
    bindings->clipView = cairoCFerBind_clipView;
    bindings->endView = cairoCFerBind_endView;
    bindings->updateWindow = cairoCFerBind_updateWindow;
    bindings->clearWindow = cairoCFerBind_clearWindow;
    bindings->redrawWindow = cairoCFerBind_redrawWindow;
    bindings->windowDpi = cairoCFerBind_windowDpi;
    bindings->resizeWindow = cairoCFerBind_resizeWindow;
    bindings->showWindow = cairoCFerBind_showWindow;
    bindings->saveWindow = cairoCFerBind_saveWindow;
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
    bindings->drawText = cairoCFerBind_drawText;

    /* Create the instance-specific data structure */
    bindings->instancedata = (CairoCFerBindData *) PyMem_Malloc(sizeof(CairoCFerBindData));
    if ( bindings->instancedata == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createWindow: "
                            "Out of memory for a CairoCFerBindData structure");
        PyMem_Free(bindings);
        return NULL;
    }
    /* Initialize everything to zero */
    memset(bindings->instancedata, 0, sizeof(CairoCFerBindData));

    /* Set non-zero default values */
    instdata = (CairoCFerBindData *) bindings->instancedata;
    /* Default Cairo surface type */
    instdata->imageformat = CCFBIF_REC;
    /* image size and minimum allowed value */
    instdata->imagewidth = 840;
    instdata->imageheight = 720;
    instdata->minsize = 128;
    /* Assign the line width scaling factor */
    instdata->widthfactor = 1.0;
    /* default clear color of opaque white */
    instdata->lastclearcolor.id = CCFBColorId;
    instdata->lastclearcolor.redfrac = 1.0;
    instdata->lastclearcolor.greenfrac = 1.0;
    instdata->lastclearcolor.bluefrac = 1.0;
    instdata->lastclearcolor.opaquefrac = 1.0;

    return bindings;
}

