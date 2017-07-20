#ifndef _FERMEM_H_
#define _FERMEM_H_

#include <stdlib.h> /* for size_t */

void *FerMem_Malloc(size_t size);
void *FerMem_Realloc(void *ptr, size_t size);
void  FerMem_Free(void *ptr);

#endif
