/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
/* Write error message directly to grdelerrmsg */
#include "grdel.h"

/*
 * Creates and returns a pointer to a Cario instance of a
 * CFerBind struct 
 */
CFerBind *cairoCFerBind_createWindow(void)
{
    CFerBind *bindings;
    int k;

    bindings = (CFerBind *) PyMem_Malloc(sizeof(CFerBind));
    if ( bindings == NULL ) {
        sprintf(grdelerrmsg, "Out of memory for a %s CFerBind bindings", CairoCFerBindName);
        return NULL;
    }
    bindings->enginename = CairoCFerBindName;

    bindings->instancedata = (CairoCFerBindData *) PyMem_Malloc(sizeof(CairoCFerBindData));
    if ( bindings->instancedata == NULL ) {
        sprintf(grdelerrmsg, "Out of memory for a %s CFerBind bindings", CairoCFerBindName);
        PyMem_Free(bindings);
        return NULL;
    }

    bindings->deleteWindow = cairoCFerBind_deleteWindow;
    bindings->setAntialias = cairoCFerBind_setAntialias;
    bindings->beginView = cairoCFerBind_beginView;
    bindings->clipView = cairoCFerBind_clipView;
    bindings->endView = cairoCFerBind_endView;
    bindings->updateWindow = cairoCFerBind_updateWindow;
    bindings->clearWindow = cairoCFerBind_clearWindow;
    bindings->windowDpi = cairoCFerBind_windowDpi;
    bindings->resizeWindow = cairoCFerBind_resizeWindow;
    bindings->showWindow = cairoCFerBind_showWindow;
    bindings->saveWindow = cairoCFerBind_saveWindow;
    bindings->createColor = cairoCFerBind_createColor;
    bindings->deleteColor = cairoCFerBind_deleteColor;
    bindings->createFont = cairoCFerBind_createFont;
    bindings->deleteFont = cairoCFerBind_deleteFont;
    bindings->createPen = cairoCFerBind_createPen;
    bindings->deletePen = cairoCFerBind_deletePen;
    bindings->createBrush = cairoCFerBind_createBrush;
    bindings->deleteBrush = cairoCFerBind_deleteBrush;
    bindings->createSymbol = cairoCFerBind_createSymbol;
    bindings->deleteSymbol = cairoCFerBind_deleteSymbol;
    bindings->drawMultiline = cairoCFerBind_drawMultiline;
    bindings->drawPoints = cairoCFerBind_drawPoints;
    bindings->drawPolygon = cairoCFerBind_drawPolygon;
    bindings->drawRectangle = cairoCFerBind_drawRectangle;
    bindings->drawMulticoloredRectangle = cairoCFerBind_drawMulticoloredRectangle;
    bindings->drawText = cairoCFerBind_drawText;

    return bindings;
}

