/* Python.h should always be first */
#include <Python.h>
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Deletes (frees) any allocated resources associated
 * with this instance of the bindings, and then the 
 * bindings itself.  After calling this function, the
 * bindings should no longer be used.
 */
int cairoCFerBind_deleteWindow(CFerBind *self)
{
    PyMem_Free(self->instancedata);
    PyMem_Free(self);
    return 0;
}

