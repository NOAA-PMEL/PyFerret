/*
 *
 * Allocate c heap storage needed for the storage of Ferret mrs
 * and pass the pointer to it to an FORTRAN 90
 * routine that will save it in COMMON
 *
 * V72 6/17 *acm* For trac enhancement #767 -- dynamic coordinate storage 
 *                Following main-memory dynamic allocation methods
 *
 */

#include <stdlib.h>
#include "ferret.h"
#include "FerMem.h"

void FORTRAN(get_linemem)( int *index, long *alen, int *status )

/*
  input  - index:  Ferret line number in which to store the array pointer
  input  - alen:   array length
  output - status flag
*/

{

  double *pointer;

  pointer = (double *) FerMem_Malloc(sizeof(double) * (*alen), __FILE__, __LINE__);

  if (pointer)
    {
      FORTRAN(store_line_ptr) (index, alen, pointer);
      *status = 3;
    }
  else
    *status = 0;

  return;
}

