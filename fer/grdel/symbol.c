/*
 * Symbol objects can only be used with the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "pyferret.h"

static const char *grdelsymbolid = "GRDEL_SYMBOL";

typedef struct GDsymbol_ {
    const char *id;
    grdelType window;
    grdelType object;
} GDSymbol;


/*
 * Returns a Symbol object.
 *
 * Arguments:
 *     window: Window in which this symbol is to be used
 *     symbolname: name of the symbol (e.g., ".", "+")
 *     symbolnamelen: actual length of the symbol name
 *
 * Returns a pointer to the symbol object created.  If an error occurs,
 * NULL is returned and grdelerrmsg contains an explanatory message.
 */
grdelType grdelSymbol(grdelType window, const char *symbolname,
                      int symbolnamelen)
{
    const BindObj *bindings;
    GDSymbol *symbol;

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "grdelSymbol: window argument is not "
                            "a grdel Window");
        return NULL;
    }

    symbol = (GDSymbol *) PyMem_Malloc(sizeof(GDSymbol));
    if ( symbol == NULL ) {
        strcpy(grdelerrmsg, "grdelSymbol: out of memory for a new Symbol");
        return NULL;
    }

    symbol->id = grdelsymbolid;
    symbol->window = window;
    if ( bindings->cferbind != NULL ) {
        symbol->object = bindings->cferbind->createSymbol(bindings->cferbind,
                                             symbolname, symbolnamelen);
        if ( symbol->object == NULL ) {
            /* grdelerrmsg already assigned */
            PyMem_Free(symbol);
            return NULL;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        symbol->object = PyObject_CallMethod(bindings->pyobject, "createSymbol",
                                  "s#", symbolname, symbolnamelen);
        if ( symbol->object == NULL ) {
            sprintf(grdelerrmsg, "grdelSymbol: error when calling the Python "
                    "binding's createSymbol method: %s", pyefcn_get_error());
            PyMem_Free(symbol);
            return NULL;
        }
    }
    else {
        strcpy(grdelerrmsg, "grdelSymbol: unexpected error, "
                            "no bindings associated with this Window");
        PyMem_Free(symbol);
        return NULL;
    }

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelSymbol created: "
            "window = %X, symbolname[0] = %c, symbol = %X\n",
            window, symbolname[0], symbol);
    fflush(debuglogfile);
#endif

    grdelerrmsg[0] = '\0';
    return symbol;
}

/*
 * Verifies symbol is a grdel Symbol.  If window is not NULL,
 * also verifies symbol can be used with this Window.
 * Returns a pointer to the graphic engine's symbol object
 * if successful.  Returns NULL if there is a problem.
 */
grdelType grdelSymbolVerify(grdelType symbol, grdelType window)
{
    GDSymbol *mysymbol;

    if ( symbol == NULL )
        return NULL;
    mysymbol = (GDSymbol *) symbol;
    if ( mysymbol->id != grdelsymbolid )
        return NULL;
    if ( (window != NULL) && (mysymbol->window != window) )
        return NULL;
    return mysymbol->object;
}

/*
 * Delete a Symbol created by grdelSymbol
 *
 * Arguments:
 *     symbol: Symbol to be deleted
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelSymbolDelete(grdelType symbol)
{
    const BindObj *bindings;
    GDSymbol *mysymbol;
    grdelBool success;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelSymbolDelete called: "
            "symbol = %X\n", symbol);
    fflush(debuglogfile);
#endif

    if ( grdelSymbolVerify(symbol, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelSymbolDelete: symbol argument is not "
                            "a grdel Symbol");
        return 0;
    }
    mysymbol = (GDSymbol *) symbol;

    grdelerrmsg[0] = '\0';
    success = 1;

    bindings = grdelWindowVerify(mysymbol->window);
    if ( bindings->cferbind != NULL ) {
        success = bindings->cferbind->deleteSymbol(bindings->cferbind,
                                                   mysymbol->object);
        /* if there was a problem, grdelerrmsg is already assigned */
    }
    else if ( bindings->pyobject != NULL ) {
        /* "N" - steals the reference to this symbol object */
        result = PyObject_CallMethod(bindings->pyobject, "deleteSymbol",
                                     "N", (PyObject *) mysymbol->object);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelSymbolDelete: error when calling the Python "
                    "binding's deleteSymbol method: %s", pyefcn_get_error());
            success = 0;
        }
        else
            Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelSymbolDelete: unexpected error, "
                            "no bindings associated with this Window");
        success = 0;
    }

    /* regardless of success, free this Symbol */
    mysymbol->id = NULL;
    mysymbol->window = NULL;
    mysymbol->object = NULL;
    PyMem_Free(mysymbol);

    return success;
}

/*
 * Creates a Symbol object.
 *
 * Input Arguments:
 *     window: Window in which this symbol is to be used
 *     symbolname: name of the symbol (e.g., ".", "+")
 *     symbolnamelen: actual length of the symbol name
 * Output Arguments:
 *     symbol: the created symbol object, or zero if failure.
 *             Use fgderrmsg_ to retrieve the error message.
 */
void fgdsymbol_(void **symbol, void **window, char *symbolname, int *namelen)
{
    grdelType mysymbol;

    mysymbol = grdelSymbol(*window, symbolname, *namelen);
    *symbol = mysymbol;
}

/*
 * Deletes a Symbol created by fgdsymbol_
 *
 * Input Arguments:
 *     symbol: Symbol to be deleted
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdsymboldel_(int *success, void **symbol)
{
    grdelBool result;

    result = grdelSymbolDelete(*symbol);
    *success = result;
}

