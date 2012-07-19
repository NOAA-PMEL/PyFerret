/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "EF_Util.h"

void FORTRAN(ef_set_axis_limits)(int *id_ptr, int *axis, int *lo, int *hi)
{
  ExternalFunction *ef_ptr;

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort(); 
  }
  ef_ptr->internals_ptr->axis[*axis-1].ss_lo = *lo;
  ef_ptr->internals_ptr->axis[*axis-1].ss_hi = *hi;
}

