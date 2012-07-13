
#ifdef MAC_SSIZE
typedef long ssize_t;
#endif
#include <stdlib.h>
#include <string.h>
#include "EF_Util.h"

/* 
 *  Make a copy of a null-terminated string
 *  and assign it to the output pointer.
 */
void FORTRAN(ef_put_string_ptr)(char **in_ptr, char **out_ptr)
{
  if ( *out_ptr != NULL )
     free(*out_ptr);

  *out_ptr = (char *) malloc(sizeof(char) * (strlen(*in_ptr)+1));
  if ( *out_ptr == NULL ) {
     abort();
  }
  strcpy(*out_ptr, *in_ptr);
}

