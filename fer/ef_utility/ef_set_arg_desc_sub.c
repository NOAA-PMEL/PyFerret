/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include "ferret.h"
#include "EF_Util.h"

void FORTRAN(ef_set_arg_desc_sub)(int *id_ptr, int *arg_ptr, char *text)
{
  ExternalFunction *ef_ptr;

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort();
  }
  strcpy(ef_ptr->internals_ptr->arg_desc[*arg_ptr-1], text);
}  

