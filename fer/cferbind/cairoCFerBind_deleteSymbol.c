/* Python.h should always be first */
#include <Python.h>
#include <cairo/cairo.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "FerMem.h"

/*
 * Delete a symbol object for this "Window".
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_deleteSymbol(CFerBind *self, grdelType symbol)
{
    CCFBSymbol *symbolobj;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deleteSymbol: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    symbolobj = (CCFBSymbol *) symbol;
    if ( symbolobj->id != CCFBSymbolId ) {
        strcpy(grdelerrmsg, "cairoCFerBind_deleteSymbol: unexpected error, "
                            "symbol is not CCFBSymbol struct");
        return 0;
    }

    /* Free memory for the path */
    cairo_path_destroy(symbolobj->path);

    /* Wipe the path, name, and id - to detect errors */
    memset(symbolobj->name, 0, sizeof(symbolobj->name));
    symbolobj->path = NULL;
    symbolobj->id = NULL;

    /* Free the memory */
    FerMem_Free(symbol, __FILE__, __LINE__);

    return 1;
}

