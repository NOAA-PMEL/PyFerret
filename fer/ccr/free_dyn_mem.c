/*
 * use gcc -c free_mem_ptr.c
 *
 * free c heap storage that was used by a Ferret mvar
 *
 * V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management
 */

#include <stdlib.h>
#include "ferret.h"
#include "FerMem.h"

void FORTRAN(free_dyn_mem)(double *mvar)
{
  FerMem_Free( mvar, __FILE__, __LINE__ );
  mvar = NULL;
}
