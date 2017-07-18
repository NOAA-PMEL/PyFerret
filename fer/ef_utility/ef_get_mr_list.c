#include "ferret.h"
#include "EF_Util.h"

void FORTRAN(ef_get_mr_list)(int mr_list[EF_MAX_ARGS])
{
  int i;

  if (  GLOBAL_mr_list_ptr != NULL ) {
     for (i=0; i<EF_MAX_ARGS; i++) {
        mr_list[i] = GLOBAL_mr_list_ptr[i];
     }
  }
  else {
     for (i=0; i<EF_MAX_ARGS; i++) {
        mr_list[i] = 0;  /* flag that mr_list is not available */
     }
  }
}

