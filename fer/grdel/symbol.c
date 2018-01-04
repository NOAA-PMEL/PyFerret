/*
 * Symbol objects can only be used with the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "pyferret.h"
#include "FerMem.h"

static const char *grdelsymbolid = "GRDEL_SYMBOL";

typedef struct GDsymbol_ {
    const char *id;
    grdelType window;
    grdelType object;
} GDSymbol;


/*
 * Returns a Symbol object.
 * 
 * If numpts is less than one, or if ptsx or ptsy is NULL, the symbol name 
 * must already be known, either as a pre-defined symbol or from a previous 
 * call to this function.
 *
 * If numpts is greater than zero and ptsx and ptsy are not NULL, the 
 * arguments ptsx and ptsy are X- and Y-coordinates that define the symbol 
 * as multiline subpaths in a [-50,50] square.  The location of the point 
 * this symbol represents will be at the center of the square.  An invalid 
 * coordinate (outside [-50,50]) will terminate the current subpath, and 
 * the next valid coordinate will start a new subpath.  If the start and 
 * end of a subpath coincide, the path will be closed.  This definition 
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
grdelType grdelSymbol(grdelType window, const char *symbolname, int symbolnamelen,
                      const float ptsx[], const float ptsy[], int numpts)
{
    const BindObj *bindings;
    GDSymbol *symbol;
    PyObject *ptstuple;
    PyObject *pairtuple;
    PyObject *fltobj;
    int       k;

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "grdelSymbol: window argument is not "
                            "a grdel Window");
        return NULL;
    }

    symbol = (GDSymbol *) FerMem_Malloc(sizeof(GDSymbol), __FILE__, __LINE__);
    if ( symbol == NULL ) {
        strcpy(grdelerrmsg, "grdelSymbol: out of memory for a new Symbol");
        return NULL;
    }

    symbol->id = grdelsymbolid;
    symbol->window = window;
    if ( bindings->cferbind != NULL ) {
        symbol->object = bindings->cferbind->createSymbol(bindings->cferbind, 
                                   symbolname, symbolnamelen, ptsx, ptsy, numpts);
        if ( symbol->object == NULL ) {
            /* grdelerrmsg already assigned */
            FerMem_Free(symbol, __FILE__, __LINE__);
            return NULL;
        }
    }
    else if ( bindings->pyobject != NULL ) {
        if ( (numpts > 0) && (ptsx != NULL) && (ptsy != NULL) ) {
            ptstuple = PyTuple_New( (Py_ssize_t) numpts );
            if ( ptstuple == NULL ) {
                PyErr_Clear();
                strcpy(grdelerrmsg, "grdelSymbol: problems creating "
                                    "a Python tuple");
                FerMem_Free(symbol, __FILE__, __LINE__);
                return 0;
            }
            for (k = 0; k < numpts; k++) {
                /* pair of floats per point */
                pairtuple = PyTuple_New( (Py_ssize_t) 2 );
                if ( pairtuple == NULL ) {
                    PyErr_Clear();
                    strcpy(grdelerrmsg, "grdelSymbol: problems creating "
                                        "a Python tuple");
                    Py_DECREF(ptstuple);
                    FerMem_Free(symbol, __FILE__, __LINE__);
                    return 0;
                }
                /* X coordinate of this point */
                fltobj = PyFloat_FromDouble((double) ptsx[k]);
                if ( fltobj == NULL ) {
                    PyErr_Clear();
                    strcpy(grdelerrmsg, "grdelSymbol: problems creating "
                                        "a Python float");
                    Py_DECREF(pairtuple);
                    Py_DECREF(ptstuple);
                    FerMem_Free(symbol, __FILE__, __LINE__);
                    return 0;
                }
                /* PyTuple_SET_ITEM steals the reference to fltobj */
                PyTuple_SET_ITEM(pairtuple, (Py_ssize_t) 0, fltobj);
                /* Y coordinate of this point */
                fltobj = PyFloat_FromDouble((double) ptsy[k]);
                if ( fltobj == NULL ) {
                    PyErr_Clear();
                    strcpy(grdelerrmsg, "grdelSymbol: problems creating "
                                        "a Python float");
                    Py_DECREF(pairtuple);
                    Py_DECREF(ptstuple);
                    FerMem_Free(symbol, __FILE__, __LINE__);
                    return 0;
                }
                /* PyTuple_SET_ITEM steals the reference to fltobj */
                PyTuple_SET_ITEM(pairtuple, (Py_ssize_t) 1, fltobj);
                /* Add this pair to the tuple of points */
                PyTuple_SET_ITEM(ptstuple, (Py_ssize_t) k, pairtuple);
            }
        }
        else {
            ptstuple = Py_None;
            Py_INCREF(Py_None);
        }
        /*
         * Call the createSymbol method of the bindings instance.
         * Using 'N' to steal the reference to xtuple and to ytuple.
         */
        symbol->object = PyObject_CallMethod(bindings->pyobject, "createSymbol",
                                  "s#N", symbolname, symbolnamelen, ptstuple);
        if ( symbol->object == NULL ) {
            sprintf(grdelerrmsg, "grdelSymbol: error when calling the Python "
                    "binding's createSymbol method: %s", pyefcn_get_error());
            FerMem_Free(symbol, __FILE__, __LINE__);
            return NULL;
        }
    }
    else {
        strcpy(grdelerrmsg, "grdelSymbol: unexpected error, "
                            "no bindings associated with this Window");
        FerMem_Free(symbol, __FILE__, __LINE__);
        return NULL;
    }

#ifdef GRDELDEBUG
    {
        char *name = (char *) FerMem_Malloc(symbolnamelen+1, __FILE__, __LINE__);
        strncpy(name, symbolname, symbolnamelen);
        name[symbolnamelen] = '\0';
        fprintf(debuglogfile, "grdelSymbol created: "
                "window = %p, symbolname = %s, symbol = %p\n",
                window, name, symbol);
        FerMem_Free(name, __FILE__, __LINE__);
    }
    if ( (numpts > 0) && (ptsx != NULL) && (ptsy != NULL) ) {
        fprintf(debuglogfile, "    from points: (%#6.2f,%#6.2f)", ptsx[0], ptsy[0]);
        for (k = 1; k < numpts; k++)
            fprintf(debuglogfile, ", (%#6.2f, %#6.2f)", ptsx[k], ptsy[k]);
        fputc('\n', debuglogfile);
    }
    fflush(debuglogfile);
#endif

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

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelSymbolDelete called: "
            "symbol = %p\n", symbol);
    fflush(debuglogfile);
#endif

    if ( grdelSymbolVerify(symbol, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelSymbolDelete: symbol argument is not "
                            "a grdel Symbol");
        return 0;
    }
    mysymbol = (GDSymbol *) symbol;

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
    FerMem_Free(mysymbol, __FILE__, __LINE__);

    return success;
}

/*
 * Creates a Symbol object.
 *
 * Input Arguments:
 *     window: Window in which this symbol is to be used
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
 * Output Arguments:
 *     symbol: the created symbol object, or zero if failure.
 *             Use fgderrmsg_ to retrieve the error message.
 */
void fgdsymbol_(void **symbol, void **window, char *symbolname, int *namelen, 
                float ptsx[], float ptsy[], int *numpts)
{
    grdelType mysymbol;

    mysymbol = grdelSymbol(*window, symbolname, *namelen, ptsx, ptsy, *numpts);
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

