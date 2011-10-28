/*
 * Symbol objects can only be used with the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "pyferret.h"

static const char *grdelsymbolid = "GRDEL_SYMBOL";

typedef struct GDsymbol_ {
    const char *id;
    grdelType window;
    PyObject *object;
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
    PyObject *bindings;
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
    symbol->object = PyObject_CallMethod(bindings, "createSymbol", "s#",
                                         symbolname, symbolnamelen);
    if ( symbol->object == NULL ) {
        sprintf(grdelerrmsg, "grdelSymbol: error when calling "
                "the binding's createSymbol method: %s", pyefcn_get_error());
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
PyObject *grdelSymbolVerify(grdelType symbol, grdelType window)
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
    GDSymbol *mysymbol;
    PyObject *bindings;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelSymbolDelete called: "
            "symbol = %X\n", symbol);
    fflush(debuglogfile);
#endif

    if ( grdelSymbolVerify(symbol, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelSymbolDelete: symbol argument is not "
                            "a grdel Symbol");
        return (grdelBool) 0;
    }
    mysymbol = (GDSymbol *) symbol;

    bindings = grdelWindowVerify(mysymbol->window);
    /* "N" - steals the reference to this symbol object */
    result = PyObject_CallMethod(bindings, "deleteSymbol", "N",
                                 mysymbol->object);
    if ( result == NULL )
        sprintf(grdelerrmsg, "grdelSymbolDelete: error when calling "
                "the binding's deleteSymbol method: %s", pyefcn_get_error());
    else
        Py_DECREF(result);

    mysymbol->id = NULL;
    mysymbol->window = NULL;
    mysymbol->object = NULL;
    PyMem_Free(mysymbol);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
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

