#include <stdio.h>
#include <stdlib.h>
#include "FerMem.h"

/*
 * Functions that just forward to the appropriate malloc, realloc, and free functions.
 * This is done to remove differences in code for Ferret and PyFerret in that all the
 * differences should be contained in this files.  This also allows an easy method of 
 * "old-fashioned" debugging (print statements) of memory issues with the appropriate 
 * compiler flag.  Optimization should inline these functions, so the extra layer of 
 * function calls should not cost time in optimized code.
 *
 * A calloc equivalent is intentionally not provided.
 * To replicate calloc: ptr = FerMem_Malloc(size); memset(ptr, 0, size);
 *
 * 07/2017 *KMS* - Initial version
 */

#ifdef MEMORYDEBUG
#define DEBUGFILENAME "memorydebug.txt"
static void writedebug(char *msg) {
    FILE *debugfile = fopen(DEBUGFILENAME, "a");
    if ( debugfile == NULL ) {
        perror("Unable to open for appending memory debug file " DEBUGFILENAME);
        return;
    }
    fputs(msg, debugfile);
    fclose(debugfile);
}
#endif

void *FerMem_Malloc(size_t size) {
    void *result = malloc(size);

#ifdef MEMORYDEBUG
    char *char_result = (char *)result;
    size_t k;
    char msg[256];

    /* initialize to non-zero junk to catch uninitialized memory usage */
    for (k = 0; k < size; k++)
        char_result[k] = 0x6B;
    sprintf(msg, "%p : 1 : memory malloc allocated for %u bytes\n", result, size);
    writedebug(msg);
#endif

    return result;
}

void *FerMem_Realloc(void *ptr, size_t size) {
    void *newptr;

#ifdef MEMORYDEBUG
    char msg[256];
    sprintf(msg, "%p : 2 : memory to be realloc freed\n", ptr);
#endif

    newptr = realloc(ptr, size);

#ifdef MEMORYDEBUG
    sprintf(msg, "%p : 3 : memory realloc allocated for %u bytes\n", newptr, size);
    writedebug(msg);
#endif

    return newptr;
}

void FerMem_Free(void *ptr) {

#ifdef MEMORYDEBUG
    char msg[256];
    sprintf(msg, "%p : 4 : memory to be freed\n", ptr);
    writedebug(msg);
#endif

    free(ptr);
}

