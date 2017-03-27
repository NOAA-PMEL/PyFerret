/*
 * use gcc -c free_mem_ptr.c
 *
 * free c heap storage that was used by a Ferret mvar
 *
 * V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management
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


void FORTRAN(free_dyn_mem) ( double *mvar )

/*
  input  - mvar: pointer to memory allocated to Ferret variable, "mvar"
*/

{

  free(mvar);

  return;
}
