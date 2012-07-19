/* Make sure Python.h is included first */
#include <Python.h>
#include "EF_Util.h"

void FORTRAN(ef_get_cx_list)(int cx_list[EF_MAX_ARGS])
{
  int i;

  for (i=0; i<EF_MAX_ARGS; i++) {
     cx_list[i] = GLOBAL_cx_list_ptr[i];
  }
}

