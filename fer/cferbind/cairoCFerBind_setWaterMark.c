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

grdelBool cairoCFerBind_setWaterMark(CFerBind *self, const char filename[],
                                     int len_filename)
{
    int  j, k;
    char fmtext[8];
    CCFBImageFormat imageformat;
    CairoCFerBindData *instdata;
    CCFBPicture *delpic;

    /* Sanity checks - this should NOT be called by the PyQtCairo engine */
    if ( self->enginename != CairoCFerBindName ) {
        strcpy(grdelerrmsg, "cairoCFerBind_setWaterMark: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }

    strcpy(grdelerrmsg, "cairoCFerBind_setWaterMark: not implemented");

    return 0;
}
