/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "EF_Util.h"

/*
 * Set the number of work arrays requested by a function.
 */
void FORTRAN(ef_set_num_work_arrays)(int *id_ptr, int *num_arrays)
{
  ExternalFunction *ef_ptr;

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort();
  }
  ef_ptr->internals_ptr->num_work_arrays = *num_arrays;
}

