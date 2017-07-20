/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include "ferret.h"
#include "EF_Util.h"
#include "FerMem.h"

/* 
 *  Make a copy of a null-terminated string
 *  and assign it to the output pointer.
 */
void FORTRAN(ef_put_string_ptr)(char **in_ptr, char **out_ptr)
{
  if ( *out_ptr != NULL )
     FerMem_Free(*out_ptr);

  *out_ptr = (char *) FerMem_Malloc(sizeof(char) * (strlen(*in_ptr)+1));
  if ( *out_ptr == NULL ) {
     abort();
  }
  strcpy(*out_ptr, *in_ptr);
}

