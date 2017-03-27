/*
 * use gcc -c nullify_mr.c
 *
 * set the F90 array pointer to null
 *
 * V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management
 *
 */


#include <stdlib.h>
/*  the relevant definition of "FORTRAN" pulled from
 *   #include "../common/ferret.h"
 */
#ifndef FORTRAN
#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif
#endif


void FORTRAN(nullify_mr)( int *mr )

/*
  input  - mr:  Ferret mr index at which to store the array pointer
*/

{

  double *nul_ptr;

  nul_ptr = (double *) NULL;

  FORTRAN(store_nul_mr_ptr) (mr, nul_ptr);

  return;
}
