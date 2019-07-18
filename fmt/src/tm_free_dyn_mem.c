/*
 * use gcc -c tm_dyn_free_mem.c
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


void FORTRAN(tm_free_dyn_mem) ( double *lm )

/*
  input  - lm: pointer to memory allocated to line storage, "lm"
*/

{

  free(lm);

  return;
}
