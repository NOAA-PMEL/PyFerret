/*
 * use gcc -c nullify_linemem.c
 *
 * set the F90 array pointer to null
 *
 * V72 6/17 *acm* For trac enhancement #767 -- dynamic coordinate storage 
 *
 */


#include <Python.h> /* make sure Python.h is first */
#include <stdlib.h>
#include "ferret.h"


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
