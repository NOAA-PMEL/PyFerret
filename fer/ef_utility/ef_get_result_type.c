#include <stdlib.h>
#include "EF_Util.h"

void FORTRAN(ef_get_result_type)(int *id_ptr, int *type)
{
  ExternalFunction *ef_ptr;

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort();
  }
  *type = ef_ptr->internals_ptr->return_type;
}  

