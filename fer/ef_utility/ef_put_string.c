/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "ferret.h"
#include "FerMem.h"

/* 
 *  Make a copy of a string up to a given length
 *  and assign it to the output pointer.
 */
void FORTRAN(ef_put_string)(char* text, int* inlen, char** out_ptr)
{
   int i;

   if ( *out_ptr != NULL )
      FerMem_Free(*out_ptr);

   *out_ptr = (char *) FerMem_Malloc(sizeof(char) * (*inlen+1));
   if ( *out_ptr == NULL ) {
      abort();
   }
   for (i=0; i<*inlen; i++) {
      (*out_ptr)[i] = text[i];
   }
   (*out_ptr)[*inlen] = '\0';    /* null-terminate the stored string */
}

