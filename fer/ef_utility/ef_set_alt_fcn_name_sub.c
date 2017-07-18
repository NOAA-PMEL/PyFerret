#include <stdlib.h>
#include <string.h>
#include "ferret.h"
#include "EF_Util.h"

void FORTRAN(ef_set_alt_fcn_name_sub)(int *id_ptr, char *text)
{
   ExternalFunction *ef_ptr;

   ef_ptr = ef_ptr_from_id_ptr(id_ptr);
   if ( ef_ptr == NULL ) {
      abort();
   }
   strcpy(ef_ptr->internals_ptr->alt_fcn_name, text);
}  

