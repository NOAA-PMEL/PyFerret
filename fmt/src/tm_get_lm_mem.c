#include <Python.h> /* make sure Python.h is first */
#include <stdlib.h>
#include <stdlib.h>
#include "FerMem.h"
#include "fmtprotos.h"


void FORTRAN(tm_get_lm_mem)( int *index, long *alen, int *status )

/*
  input  - index:  Ferret ws index at which to store the array pointer
  input  - alen:   array length
*/

{

  double *pointer;

  pointer = (double *) FerMem_Malloc(sizeof(double) * (*alen), __FILE__, __LINE__);

  if (pointer)
    {
      FORTRAN(tm_store_lm_ptr) (index, alen, pointer);
      *status = 3;
    }
  else
    *status = 0;

  return;
}
