#include <Python.h> /* make sure Python.h is first */
#include <stdlib.h>
#include "fmtprotos.h"

void FORTRAN(tm_nullify_lm)( int *lm )

/*
  input  - lm:  line memory index at which to store the array pointer
*/

{

  double *nul_ptr;

  nul_ptr = (double *) NULL;

  FORTRAN(tm_store_nul_lm_ptr) (lm, nul_ptr);

  return;
}
