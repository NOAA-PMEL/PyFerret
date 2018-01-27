/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "FerMem.h"

/* Instantiate the global value */
const char *CCFBSymbolId = "CCFBSymbolId";

/*
 * Create a Symbol object for this "Window".
 * 
 * If numpts is less than one, or if ptsx or ptsy is NULL, the symbol name 
 * must already be known, either as a pre-defined symbol or from a previous 
 * call to this function.
 *
 * Currently pre-defined symbols are:
 *     "." (period) - small filled circle
 *     "o" (lowercase oh) - unfilled circle
 *     "+" (plus) - plus
 *     "x" (lowercase ex) - ex
 *     "*" (asterisk) - asterisk
 *     "^" (caret) - unfilled triangle
 *     "#" (pound sign) - unfilled square
 *
 * If numpts is greater than zero and ptsx and ptsy are not NULL, the 
 * arguments ptsx and ptsy are X- and Y-coordinates that define the symbol 
 * as multiline subpaths in a [-50,50] square.  The location of the point 
 * this symbol represents will be at the center of the square.  An invalid 
 * coordinate (outside [-50,50]) will terminate the current subpath, and 
 * the next valid coordinate will start a new subpath.  This definition 
 * will replace an existing symbol with the given name.
 *
 * Arguments:
 *     window: Window in which this symbol is to be used
 *     symbolname: name of the symbol
 *     symbolnamelen: actual length of the symbol name
 *     ptsx: vertex X-coordinates 
 *     ptsy: vertex Y-coordinates 
 *     numpts: number of vertices
 *
 * Returns a pointer to the symbol object created.  If an error occurs,
 * NULL is returned and grdelerrmsg contains an explanatory message.
 */
grdelType cairoCFerBind_createSymbol(CFerBind *self, const char *symbolname, int namelen,
                        const float ptsx[], const float ptsy[], int numpts, grdelBool fill)
{
    CairoCFerBindData *instdata;
    CCFBSymbol *symbolobj;
    cairo_surface_t *pathsurface;
    cairo_t *pathcontext;
    int      somethingdrawn;
    int      newstart;
    int      laststart;
    int      lastend;
    int      k;
    double   xval;
    double   yval;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createSymbol: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;
    if ( instdata->context == NULL ) {
        /* Create the Cairo Surface and Context if they do not exist */
        if ( ! cairoCFerBind_createSurface(self) ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }

    /* Create the symbol object */
    symbolobj = (CCFBSymbol *) FerMem_Malloc(sizeof(CCFBSymbol), __FILE__, __LINE__);
    if ( symbolobj == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createSymbol: "
                            "out of memory for a CCFBSymbol structure");
        return NULL;
    }
    symbolobj->id = CCFBSymbolId;

    /* Copy the symbol name */
    if ( namelen >= sizeof(symbolobj->name) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createSymbol: symbol name too long");
        FerMem_Free(symbolobj, __FILE__, __LINE__);
        return NULL;
    }
    strncpy(symbolobj->name, symbolname, namelen);
    symbolobj->name[namelen] = '\0';

    /* Create a 100x100 (pixel or point) surface matching that of the actual drawing surface */
    pathsurface = cairo_surface_create_similar(instdata->surface, 
                        cairo_surface_get_content(instdata->surface), 100, 100);
    if ( cairo_surface_status(pathsurface) != CAIRO_STATUS_SUCCESS ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createSymbol: unable to create surface for symbol");
        cairo_surface_destroy(pathsurface);
        FerMem_Free(symbolobj, __FILE__, __LINE__);
        return NULL;
    }

    /* Create a context to draw to this surface */
    pathcontext = cairo_create(pathsurface);
    if ( cairo_status(pathcontext) != CAIRO_STATUS_SUCCESS ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createSymbol: unable to create context for drawing the symbol");
        cairo_destroy(pathcontext);
        cairo_surface_destroy(pathsurface);
        FerMem_Free(symbolobj, __FILE__, __LINE__);
        return NULL;
    }
    cairo_set_line_width(pathcontext, 15.0);
    cairo_set_dash(pathcontext, NULL, 0, 0.0);
    cairo_set_line_cap(pathcontext, CAIRO_LINE_CAP_SQUARE);
    cairo_set_line_join(pathcontext, CAIRO_LINE_JOIN_BEVEL);
    cairo_translate(pathcontext, 50.0, 50.0);

    /* If points are given, always create the symbol from these points */
    if ( (numpts > 0) && (ptsx != NULL) && (ptsy != NULL) ) {
        somethingdrawn = 0;
        newstart = 1;
        laststart = -1;
        lastend = -1;
        for (k = 0; k < numpts; k++) {
            xval = (double) (ptsx[k]);
            /* flip so positive y is up */
            yval = -1.0 * (double) (ptsy[k]);
            if ( (xval < -50.0) || (xval > 50.0) || (yval < -50.0) || (yval > 50.0) ) {
                /* check if the current subpath should be closed */
                if ( (laststart >= 0) && (lastend > laststart) && 
                     (fabs(ptsx[lastend] - ptsx[laststart]) < 0.001) &&
                     (fabs(ptsy[lastend] - ptsy[laststart]) < 0.001) ) {
	            cairo_close_path(pathcontext);
                }
                /* and end the current subpath */
                newstart = 1;
                laststart = -1;
                lastend = -1;
            }
            else if ( newstart ) {
                /* start a new subpath */
                cairo_move_to(pathcontext, xval, yval);
                newstart = 0;
                laststart = k;
            }
            else {
                /* continue the current subpath */
                cairo_line_to(pathcontext, xval, yval);
                lastend = k;
                somethingdrawn = 1;
            }
        }
        if ( ! somethingdrawn ) {
            strcpy(grdelerrmsg, "cairoCFerBind_createSymbol: symbol definition does not contain any drawn lines");
            cairo_destroy(pathcontext);
            cairo_surface_destroy(pathsurface);
            FerMem_Free(symbolobj, __FILE__, __LINE__);
            return NULL;
        }
        symbolobj->filled = fill;
    }
    else if ( strcmp(".", symbolobj->name) == 0 ) {
        cairo_new_path(pathcontext);
        cairo_arc(pathcontext, 0.0, 0.0, 10.0, 0.0, 2.0 * M_PI);
	cairo_close_path(pathcontext);
        symbolobj->filled = 1;
    }
    else if ( strcmp("o", symbolobj->name) == 0 ) {
        cairo_new_path(pathcontext);
        cairo_arc(pathcontext, 0.0, 0.0, 40.0, 0.0, 2.0 * M_PI);
        cairo_close_path(pathcontext);
        symbolobj->filled = 0;
    }
    else if ( strcmp("+", symbolobj->name) == 0 ) {
        cairo_new_path(pathcontext);
        cairo_move_to(pathcontext,   0.0, -40.0);
        cairo_line_to(pathcontext,   0.0,  40.0);
        cairo_move_to(pathcontext, -40.0,   0.0);
        cairo_line_to(pathcontext,  40.0,   0.0);
        symbolobj->filled = 0;
    }
    else if ( strcmp("x", symbolobj->name) == 0 ) {
        cairo_new_path(pathcontext);
        cairo_move_to(pathcontext, -30.0, -30.0);
        cairo_line_to(pathcontext,  30.0,  30.0);
        cairo_move_to(pathcontext, -30.0,  30.0);
        cairo_line_to(pathcontext,  30.0, -30.0);
        symbolobj->filled = 0;
    }
    else if ( strcmp("*", symbolobj->name) == 0 ) {
        cairo_new_path(pathcontext);
        cairo_move_to(pathcontext,     0.0, -40.0);
        cairo_line_to(pathcontext,     0.0,  40.0);
        cairo_move_to(pathcontext, -34.641, -20.0);
        cairo_line_to(pathcontext,  34.641,  20.0);
        cairo_move_to(pathcontext, -34.641,  20.0);
        cairo_line_to(pathcontext,  34.641, -20.0);
        symbolobj->filled = 0;
    }
    else if ( strcmp("^", symbolobj->name) == 0 ) {
        cairo_new_path(pathcontext);
        cairo_move_to(pathcontext, -40.0,  30.0);
        cairo_line_to(pathcontext,   0.0, -39.282);
        cairo_line_to(pathcontext,  40.0,  30.0);
        cairo_close_path(pathcontext);
        symbolobj->filled = 0;
    }
    else if ( strcmp("#", symbolobj->name) == 0 ) {
        cairo_new_path(pathcontext);
        cairo_rectangle(pathcontext, -35.0, -35.0, 70.0, 70.0);
        symbolobj->filled = 0;
    }
    else {
        sprintf(grdelerrmsg, "cairoCFerBind_createSymbol: unknown symbol '%s'", symbolobj->name);
        cairo_destroy(pathcontext);
        cairo_surface_destroy(pathsurface);
        FerMem_Free(symbolobj, __FILE__, __LINE__);
        return NULL;
    }

    /* Get the path object, but with all curves converted to multilines */
    symbolobj->path = cairo_copy_path_flat(pathcontext);
    if ( symbolobj->path->data == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createSymbol: unable to generate path object for symbol");
        cairo_path_destroy(symbolobj->path);
        cairo_destroy(pathcontext);
        cairo_surface_destroy(pathsurface);
        FerMem_Free(symbolobj, __FILE__, __LINE__);
        return NULL;
    }

    /* No longer need the context and the surface used to create this path */
    cairo_destroy(pathcontext);
    cairo_surface_destroy(pathsurface);

    return symbolobj;
}

