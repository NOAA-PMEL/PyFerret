/*
 * use gcc -c tm_nullify_lm.c
 *
 * set the F90 array pointer to null
 *
 * V72 1/17 *sh* -- dynamic line memory management
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
