/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include "ferret.h"
#include "EF_Util.h"

void FORTRAN(ef_set_custom_axis_sub)(int *id_ptr, int *axis_ptr, DFTYPE *lo_ptr,
                    DFTYPE *hi_ptr, DFTYPE *del_ptr, char *text, int *modulo_ptr)
{
  ExternalFunction *ef_ptr;

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort();
  }

  strcpy(ef_ptr->internals_ptr->axis[*axis_ptr-1].unit, text);
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_lo = *lo_ptr;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_hi = *hi_ptr;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_del = *del_ptr;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].modulo = *modulo_ptr;
}  

