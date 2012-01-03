/* Python.h should always be first */
#include <Python.h>
#include <string.h>
#include "cferbind.h"
#include "grdel.h"

/* Instantiate the globals */
const char *CairoCFerBindName = "Cairo";

/*
 * Creates a CFerBind struct (bindings instance)
 * appropriately assigned for the indicated graphics engine.
 *
 * Currently, "Cairo" is the only engine supported.
 *
 * For the "Cairo" engine, the windowname, winnamelen, and
 * visible arguments are ignored as they is not applicable.
 *
 * Returns a pointer to the bindings instance if successful.
 * If an error occurs, grdelerrmsg is assigned an appropriate
 * error message and NULL is returned.
 */
CFerBind *cferbind_createWindow(char *enginename, int engnamelen,
                                char *windowname, int winnamelen, int visible)
{
    CFerBind *bindings;
    int k;

    if ( strncmp(enginename, CairoCFerBindName, strlen(CairoCFerBindName) == 0) ) {
        /* Create a bindings instance for a Cairo engine */
        bindings = cairoCFerBind_createWindow();
        return bindings;
    }

    /* Unknown CFerBind engine */
    strcpy(grdelerrmsg, "Unknown engine: ");
    if (engnamelen < 80)
        k = engnamelen;
    else
        k = 80;
    strncpy(grdelerrmsg + 16, enginename, k);
    grdelerrmsg[k + 16] = '\0';
    return NULL;
}

