/* Python.h should always be first */
#include <Python.h>
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
    CCFBSymbol *symbolobj;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createSymbol: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
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

    /* TODO: deal with generation of custom symbols */

    return symbolobj;
}

