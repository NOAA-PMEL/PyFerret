
#ifdef MAC_SSIZE
typedef long ssize_t;
#endif
#include <stdlib.h>
#include <string.h>
#include "EF_Util.h"

void FORTRAN(ef_set_desc_sub)(int *id_ptr, char *text)
{
  ExternalFunction *ef_ptr;

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort();
  }
  strcpy(ef_ptr->internals_ptr->description, text);
}  

