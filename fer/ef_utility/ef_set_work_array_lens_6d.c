/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "EF_Util.h"

/*
 * Set the requested size (in words) for a specific work array.
 */
void FORTRAN(ef_set_work_array_lens_6d)(int *id_ptr, int *iarray,
                                        int *xlen, int *ylen, int *zlen,
                                        int *tlen, int *elen, int *flen)
{
  ExternalFunction *ef_ptr;
  int array_id = *iarray - 1;      /* F to C conversion */

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort();
  }

  ef_ptr->internals_ptr->work_array_lo[array_id][0] = 1;
  ef_ptr->internals_ptr->work_array_lo[array_id][1] = 1;
  ef_ptr->internals_ptr->work_array_lo[array_id][2] = 1;
  ef_ptr->internals_ptr->work_array_lo[array_id][3] = 1;
  ef_ptr->internals_ptr->work_array_lo[array_id][4] = 1;
  ef_ptr->internals_ptr->work_array_lo[array_id][5] = 1;

  ef_ptr->internals_ptr->work_array_hi[array_id][0] = *xlen;
  ef_ptr->internals_ptr->work_array_hi[array_id][1] = *ylen;
  ef_ptr->internals_ptr->work_array_hi[array_id][2] = *zlen;
  ef_ptr->internals_ptr->work_array_hi[array_id][3] = *tlen;
  ef_ptr->internals_ptr->work_array_hi[array_id][4] = *elen;
  ef_ptr->internals_ptr->work_array_hi[array_id][5] = *flen;
}

