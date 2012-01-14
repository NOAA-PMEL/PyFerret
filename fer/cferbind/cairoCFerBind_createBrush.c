/* Python.h should always be first */
#include <Python.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/* Instatiate the global value */
const char *CCFBBrushId = "CCFBBrushId";

/*
 * Create a brush object for this "Window".
 *
 * Arguments:
 *     color     - color for the brush
 *     style     - name of the style of the brush (case-insensitive);
 *                 currently, only "solid" is supported
 *     stlen     - actual length of style
 *
 * Returns a brush object when successful.   If an error occurs,
 * grdelerrmsg is assigned an appropriate error message and NULL
 * is returned.
 */
grdelType cairoCFerBind_createBrush(CFerBind *self, grdelType color,
                                    const char *style, int stlen)
{
    CCFBColor *colorobj;
    CCFBBrush *brushobj;
    int        k;
    char       stname[16];
    int        brushtype;

    /* Sanity checks */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createBrush: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    colorobj = (CCFBColor *) color;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createBrush: unexpected error, "
                            "color is not CCFBColor struct");
        return NULL;
    }

    /*
     * Interpret the style argument.  Just assign an integer value
     * that will later be used to create the appropriate pattern.
     */
    for (k = 0; (k < stlen) && (k < 15); k++)
        stname[k] = (char) tolower(style[k]);
    stname[k] = '\0';
    if ( strcmp(stname, "solid") == 0 )
        brushtype = 0;
    else {
        sprintf(grdelerrmsg, "cairoCFerBind_createBrush: "
                             "unknown brush style of '%s'", stname);
        return NULL;
    }

    /* Create the brush object */
    brushobj = (CCFBBrush *) PyMem_Malloc(sizeof(CCFBBrush));
    if ( brushobj == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createBrush: "
                            "out of memory for a CCFBBrush structure");
        return NULL;
    }
    brushobj->id = CCFBBrushId;

    /* Copy the color structure */
    brushobj->color = *colorobj;

    /* Assign the pattern attribute, if appropriate */
    if ( brushtype == 0 ) {
        /* solid - just use the color for the source */
        brushobj->pattern = NULL;
    }
    else {
        sprintf(grdelerrmsg, "cairoCFerBind_createBrush: unexpected error, "
                             "unrecognized brushtype %d", brushtype);
        PyMem_Free(brushobj);
        return NULL;
    }

    return brushobj;
}

