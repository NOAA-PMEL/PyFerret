/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Create a symbol object for this "Window".
 *
 *     ptsx: vertices X-coordinates describing the symbol 
 *           as a multiline drawing on a [0,100] square; 
 *           only used if numpts is greater than zero
 *     ptsy: vertices Y-coordinates describing the symbol 
 *           as a multiline drawing on a [0,100] square; 
 *           only used if numpts is greater than zero
 *     numpts: number of vertices describing the symbol; 
 *           can be zero if giving a well-known symbol name
 *     symbolname: name of the symbol, either a well-known
 *           symbol name (e.g., ".") or a custom name for a 
 *           symbol created from the given vertices (e.g., "FER001")
 *     symbolnamelen: actual length of the symbol name
 *
 * Currently well-known symbol names (all single-character):
 *     '.' (period) - small filled circle
 *     'o' (lowercase oh) - unfilled circle
 *     '+' (plus) - plus
 *     'x' (lowercase ex) - ex
 *     '*' (asterisk) - asterisk
 *     '^' (caret) - unfilled triangle
 *     '#' (pound sign) - unfilled square
 *
 * Returns a sybmol object if successful.   If an error occurs,
 * grdelerrmsg is assigned an appropriate error message and NULL
 * is returned.
 */
grdelType cairoCFerBind_createSymbol(CFerBind *self, const float ptsx[], 
                                     const float ptsy[], int numpts, 
                                     const char *symbolname, int namelen)
{
    char symname[8];
    int  k;
    grdelType symbol;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createSymbol: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }

    /* null-terminate the symbol name, which should be short */
    for (k = 0; (k < 7) && (k < namelen); k++)
        symname[k] = symbolname[k];
    symname[k] = '\0';

    /* TODO: custom symbols */

    /* The symbol object is just a cast of a character value to a pointer type */
    if ( strcmp(symname, ".") == 0 )
        symbol = (grdelType) '.';
    else if ( strcmp(symname, "o") == 0 )
        symbol = (grdelType) 'o';
    else if ( strcmp(symname, "+") == 0 )
        symbol = (grdelType) '+';
    else if ( strcmp(symname, "x") == 0 )
        symbol = (grdelType) 'x';
    else if ( strcmp(symname, "*") == 0 )
        symbol = (grdelType) '*';
    else if ( strcmp(symname, "^") == 0 )
        symbol = (grdelType) '^';
    else if ( strcmp(symname, "#") == 0 )
        symbol = (grdelType) '#';
    else {
        sprintf(grdelerrmsg, "cairoCFerBind_createSymbol: "
                             "unknown symbol '%s'", symname);
        return NULL;
    }

    return symbol;
}

