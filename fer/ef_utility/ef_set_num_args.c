#include <stdlib.h>
#include <stdio.h>
#include "ferret.h"
#include "EF_Util.h"

/*
 * Set the number of args for a function.
 */
void FORTRAN(ef_set_num_args)(int *id_ptr, int *num_args)
{
   ExternalFunction *ef_ptr;
   static char err_msg[128];

   ef_ptr = ef_ptr_from_id_ptr(id_ptr);
   if ( ef_ptr == NULL ) {
      abort();
   }
   if ( *num_args > EF_MAX_ARGS ) {
      sprintf(err_msg, "Number of arguments passed to ef_set_num_args (%d) "
                       "is greater than the maximum (%d)",
                       *num_args, EF_MAX_ARGS);
      FORTRAN(ef_err_bail_out)(id_ptr, err_msg);
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   ef_ptr->internals_ptr->num_reqd_args = *num_args;
}

