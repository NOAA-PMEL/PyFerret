#include <Python.h> /* make sure Python.h is first */
#include "FerMem.h"
#include "fmtprotos.h"

void FORTRAN(tm_free_dyn_mem) ( double *lm )

/*
  input  - lm: pointer to memory allocated to line storage, "lm"
*/

{
  FerMem_Free(lm, __FILE__, __LINE__);
}
