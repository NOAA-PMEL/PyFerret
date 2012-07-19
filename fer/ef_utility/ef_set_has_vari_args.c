/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "EF_Util.h"

/*
 * Set the "variable arguments" flag for a function.
 */
void ef_set_has_vari_args_(int *id_ptr, int *has_vari_args)
{
  ExternalFunction *ef_ptr;

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort();
  }
  ef_ptr->internals_ptr->has_vari_args = *has_vari_args;
}

