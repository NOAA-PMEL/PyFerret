/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "EF_Util.h"

/*
 * Set the requested lo and hi dimensions for a specific work array.
 */
void FORTRAN(ef_set_work_array_dims_6d)(int *id_ptr, int *iarray, 
             int *xlo, int *ylo, int *zlo, int *tlo, int *elo, int *flo,
             int *xhi, int *yhi, int *zhi, int *thi, int *ehi, int *fhi)
{
  ExternalFunction *ef_ptr;
  int array_id = *iarray - 1;      /* F to C conversion */

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort();
  }

  ef_ptr->internals_ptr->work_array_lo[array_id][0] = *xlo;
  ef_ptr->internals_ptr->work_array_lo[array_id][1] = *ylo;
  ef_ptr->internals_ptr->work_array_lo[array_id][2] = *zlo;
  ef_ptr->internals_ptr->work_array_lo[array_id][3] = *tlo;
  ef_ptr->internals_ptr->work_array_lo[array_id][4] = *elo;
  ef_ptr->internals_ptr->work_array_lo[array_id][5] = *flo;

  ef_ptr->internals_ptr->work_array_hi[array_id][0] = *xhi;
  ef_ptr->internals_ptr->work_array_hi[array_id][1] = *yhi;
  ef_ptr->internals_ptr->work_array_hi[array_id][2] = *zhi;
  ef_ptr->internals_ptr->work_array_hi[array_id][3] = *thi;
  ef_ptr->internals_ptr->work_array_hi[array_id][4] = *ehi;
  ef_ptr->internals_ptr->work_array_hi[array_id][5] = *fhi;
}

