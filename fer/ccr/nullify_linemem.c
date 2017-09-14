/*
 * use gcc -c nullify_linemem.c
 *
 * set the F90 array pointer to null
 *
 * V72 6/17 *acm* For trac enhancement #767 -- dynamic coordinate storage 
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


void FORTRAN(nullify_linemem)( int *iaxis, int *line_or_edge )

/*
  input  - mr:  Ferret iaxis index at which to store the array pointer
*/

{

  double *nul_ptr;

  nul_ptr = (double *) NULL;

  FORTRAN(store_nul_line_ptr) (iaxis, line_or_edge, nul_ptr);

  return;
}
