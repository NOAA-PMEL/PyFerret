/* Make sure Python.h is included first */
#include <Python.h>
#include "EF_Util.h"

void FORTRAN(ef_get_one_val)(int *id_ptr, int *arg_ptr, DFTYPE *val_ptr)
{
  FORTRAN(ef_get_one_val_sub)(id_ptr, GLOBAL_memory_ptr, arg_ptr, val_ptr);
}

