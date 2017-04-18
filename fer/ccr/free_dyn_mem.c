/*
 * free c heap storage that was used by a Ferret mvar
 *
 * V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management
 */

#include <Python.h> /* make sure Python.h is first */
#include "ferret.h"

/*
  input  - mvar: pointer to memory allocated to Ferret variable, "mvar"
*/
void FORTRAN(free_dyn_mem) ( double *mvar )
{

  PyMem_Free(mvar);

  return;
}
