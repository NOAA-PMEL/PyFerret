/*
 * use gcc -c nullify_mr.c
 *
 * set the F90 array pointer to null
 *
 * V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management
 *
 */


#include <Python.h> /* make sure Python.h is first */
#include <stdlib.h>
#include "ferret.h"

void FORTRAN(nullify_mr)(int *mr)
/*
  input  - mr:  Ferret mr index at which to store the array pointer
*/
{
  double *nul_ptr;

  nul_ptr = (double *) NULL;

  FORTRAN(store_nul_mr_ptr)(mr, nul_ptr);

  return;
}
