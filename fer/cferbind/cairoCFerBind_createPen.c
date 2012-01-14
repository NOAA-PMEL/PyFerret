/* Python.h should always be first */
#include <Python.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/* Instatiate the global value */
const char *CCFBPenId = "CCFBPenId";

/*
 * Create a pen object for this "Window".
 *
 * Arguments:
 *     color     - color for the pen
 *     width     - unadjusted line width of the pen; this value will
 *                 be adjusted by the size of the current "View" when
 *                 used for drawing
 *     style     - name of the line style of the pen (case-insensitive);
 *                 currently, one of "solid", "dash", "dot", "dashdot"
 *     stlen     - actual length of style
 *     capstyle  - name of the cap style of the pen (case-insensitive);
 *                 currently, one of "butt", "round", "square"
 *     capstlen  - actual length of capstyle
 *     joinstyle - name of the join style of the pen (case-insensitive);
 *                 currently, one of "miter", "round", "bevel"
 *     joinstlen - actual length of joinstyle
 *
 * Returns a pen object when successful.   If an error occurs,
 * grdelerrmsg is assigned an appropriate error message and NULL
 * is returned.
 */
grdelType cairoCFerBind_createPen(CFerBind *self, grdelType color, double width,
                                  const char *style, int stlen, const char *capstyle,
                                  int capstlen, const char *joinstyle, int joinstlen)
{
    CCFBColor        *colorobj;
    CCFBPen          *penobj;
    int               k;
    char              stname[16];
    int               linetype;
    cairo_line_cap_t  captype;
    cairo_line_join_t jointype;

    /* Sanity checks */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createPen: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    colorobj = (CCFBColor *) color;
    if ( colorobj->id != CCFBColorId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createPen: unexpected error, "
                            "color is not CCFBColor struct");
        return NULL;
    }

    /*
     * When drawing, the actual pen width is always at least 1 pixel;
     * thus a width of exactly zero is always a 1 pixel wide.
     */
    if ( width < 0.0 ) {
        sprintf(grdelerrmsg, "cairoCFerBind_createPen: "
                             "invalid line width of %#.1f", width);
        return NULL;
    }

    /*
     * Interpret the style argument.  Just assign an integer value
     * that will later be used to create the appropriate dashes array.
     */
    for (k = 0; (k < stlen) && (k < 15); k++)
        stname[k] = (char) tolower(style[k]);
    stname[k] = '\0';
    if ( strcmp(stname, "solid") == 0 )
        linetype = 0;
    else if ( strcmp(stname, "dash") == 0 )
        linetype = 1;
    else if ( strcmp(stname, "dot") == 0 )
        linetype = 2;
    else if ( strcmp(stname, "dashdot") == 0 )
        linetype = 3;
    else {
        sprintf(grdelerrmsg, "cairoCFerBind_createPen: "
                             "unknown line style of '%s'", stname);
        return NULL;
    }

    /* Interpret the cap style */
    for (k = 0; (k < capstlen) && (k < 15); k++)
        stname[k] = (char) tolower(capstyle[k]);
    stname[k] = '\0';
    if ( strcmp(stname, "butt") == 0 )
        captype = CAIRO_LINE_CAP_BUTT;
    else if ( strcmp(stname, "round") == 0 )
        captype = CAIRO_LINE_CAP_ROUND;
    else if ( strcmp(stname, "square") == 0 )
        captype = CAIRO_LINE_CAP_SQUARE;
    else {
        sprintf(grdelerrmsg, "cairoCFerBind_createPen: "
                             "unknown line cap style of '%s'", stname);
        return NULL;
    }

    /* Interpret the join style */
    for (k = 0; (k < joinstlen) && (k < 15); k++)
        stname[k] = (char) tolower(joinstyle[k]);
    stname[k] = '\0';
    if ( strcmp(stname, "miter") == 0 )
        jointype = CAIRO_LINE_JOIN_MITER;
    else if ( strcmp(stname, "round") == 0 )
        jointype = CAIRO_LINE_JOIN_ROUND;
    else if ( strcmp(stname, "bevel") == 0 )
        jointype = CAIRO_LINE_JOIN_BEVEL;
    else {
        sprintf(grdelerrmsg, "cairoCFerBind_createPen: "
                             "unknown line join style of '%s'", stname);
        return NULL;
    }

    /* Create the pen object */
    penobj = (CCFBPen *) PyMem_Malloc(sizeof(CCFBPen));
    if ( penobj == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createPen: "
                            "out of memory for a CCFBPen structure");
        return NULL;
    }
    penobj->id = CCFBPenId;

    /* Copy the color structure */
    penobj->color = *colorobj;

    /* Line width */
    penobj->width = width;

    /* Assign the appropriate dashes array in units of line widths */
    switch( linetype ) {
    case 0:
        /* solid */
        penobj->numdashes = 0;
        break;
    case 1:
        /* dash */
        penobj->dashes[0] = 8.0;
        penobj->dashes[1] = 2.0;
        penobj->numdashes = 2;
        break;
    case 2:
        /* dot */
        penobj->dashes[0] = 2.0;
        penobj->dashes[1] = 2.0;
        penobj->numdashes = 2;
        break;
    case 3:
        /* dashdot */
        penobj->dashes[0] = 8.0;
        penobj->dashes[1] = 2.0;
        penobj->dashes[2] = 2.0;
        penobj->dashes[3] = 2.0;
        penobj->numdashes = 4;
        break;
    default:
        sprintf(grdelerrmsg, "cairoCFerBind_createPen: unexpected error, "
                             "linetype of %d", linetype);
        PyMem_Free(penobj);
        return NULL;
    }

    /* Assign the cap and join types */
    penobj->captype = captype;
    penobj->jointype = jointype;

    return penobj;
}

