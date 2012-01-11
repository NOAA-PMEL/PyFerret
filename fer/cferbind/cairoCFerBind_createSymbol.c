/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "grdel.h"

/*
 * Create a symbol object for this "Window".
 *
 * Currently known symbol names (all single-character):
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
grdelType cairoCFerBind_createSymbol(CFerBind *self, const char *symbolname, int namelen)
{
    char symname[8];
    int  k;
    grdelType symbol;

    /* Sanity check */
    if ( self->enginename != CairoCFerBindName ) {
        sprintf(grdelerrmsg, "cairoCFerBind_createSymbol: unexpected error, "
                             "self is not a %s CFerBind struct", CairoCFerBindName);
        return NULL;
    }

    /* null-terminate the symbol name, which should be short */
    for (k = 0; (k < 7) && (k < namelen); k++)
        symname[k] = symbolname[k];
    symname[k] = '\0';

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

