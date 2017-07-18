#include "ferret.h"
#include "EF_Util.h"

void FORTRAN(ef_get_bad_flags)(int *id_ptr, DFTYPE bad_flag[EF_MAX_ARGS], DFTYPE *bad_flag_result)
{
  int i;

  for (i=0; i<EF_MAX_ARGS; i++) {
     bad_flag[i] = GLOBAL_bad_flag_ptr[i];
  }
  *bad_flag_result = GLOBAL_bad_flag_ptr[EF_MAX_ARGS];
}

