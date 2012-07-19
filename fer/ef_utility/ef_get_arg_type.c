/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "EF_Util.h"

/*
 * Find an external function based on its integer ID and
 * return the 'arg_type' information for a particular
 * argument which tells Ferret whether an argument is a 
 * DFTYPE or a string.
 */
void FORTRAN(ef_get_arg_type)(int *id_ptr, int *iarg_ptr, int *type)
{
   ExternalFunction *ef_ptr;
   int index = *iarg_ptr - 1; 

   ef_ptr = ef_ptr_from_id_ptr(id_ptr);
   if ( ef_ptr == NULL ) {
      abort();
   }

   *type = ef_ptr->internals_ptr->arg_type[index];
}

