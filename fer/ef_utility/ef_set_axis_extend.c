/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "ferret.h"
#include "EF_Util.h"

void FORTRAN(ef_set_axis_extend)(int *id_ptr, int *arg, int *axis, int *lo, int *hi)
{
  ExternalFunction *ef_ptr=NULL;

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort();
  }
  ef_ptr->internals_ptr->axis_extend_lo[*arg-1][*axis-1] = *lo;
  ef_ptr->internals_ptr->axis_extend_hi[*arg-1][*axis-1] = *hi;
}

