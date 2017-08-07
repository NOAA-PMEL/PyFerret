#include <Python.h> /* make sure this is the first */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
static int initialized = 0;
/*
 * Add the debug message "<startptr> - <endptr> : <msg>" to the memorydebug file.
 * If <endptr> is NULL, blanks are printed instead of the NULL address.
 * For consistency, use format "%016p" to print pointers in msg.
 */
void FerMem_WriteDebugMessage(void *startptr, void *endptr, const char *msg) 
{
    FILE *debugfile;
    if ( ! initialized ) {
        initialized = 1;
        debugfile = fopen(DEBUGFILENAME, "w");
    }
    else {
        debugfile = fopen(DEBUGFILENAME, "a");
    }
    if ( debugfile == NULL ) {
        perror("Unable to open memory debug file " DEBUGFILENAME);
        exit(127);
    }
    if ( endptr != NULL ) {
        fprintf(debugfile, "%016p - %016p : %s\n", startptr, endptr, msg);
    }
    else {
        fprintf(debugfile, "%016p -                    : %s\n", startptr, msg);
    }
    fclose(debugfile);
}

#define FERMEM_BUFSIZE 8
#define FERMEM_BUFVALUE 0xAAAAAAAA

typedef struct _MemInfo_ {
    size_t  size;
    struct _MemInfo_ *next;
    size_t *headbufptr;
    size_t *tailbufptr;
    void   *memory;
    size_t  dummy[2*FERMEM_BUFSIZE];
} MemInfo;

static MemInfo *MemInfoList = NULL;

/*
 * Adds the given pointer to the linked list of memory pointers.
 * Assumes the given pointer is allocated for 
 *     'size' + sizeof(MemInfo)
 * bytes of memory; the pointer to 'size' bytes of user memory 
 * (the memory to be used as desired) is returned.
 */
static void *AddToMemInfoList(void *memplus, size_t size)
{
    MemInfo *memptr;
    int      k;
    void    *ptr;

    if ( MemInfoList == NULL ) {
        /* No list so make this memory the start of the list */
        MemInfoList = (MemInfo *) memplus;
    }
    else {
        /* Add this memory to the end of the list */
        for (memptr = MemInfoList; ; memptr = memptr->next) {
            /* First check the buffers are not corrupt */
            for (k = 0; k < FERMEM_BUFSIZE; k++) {
                if ( memptr->headbufptr[k] != FERMEM_BUFVALUE ) {
                    fprintf(stderr, "Memory underwrite detected for allocation %016p of %ld bytes\n", memptr->memory, memptr->size);
                    abort();
                }
            }
            for (k = 0; k < FERMEM_BUFSIZE; k++) {
                if ( memptr->tailbufptr[k] != FERMEM_BUFVALUE ) {
                    fprintf(stderr, "Memory overwrite detected for allocation %016p of %ld bytes\n", memptr->memory, memptr->size);
                    abort();
                }
            }
            if ( memptr->next == NULL ) {
                memptr->next = memplus;
                break;
            }
        }
    }

    /* Initialize the MemInfo block */
    memptr = (MemInfo *) memplus;
    memptr->next = NULL;
    memptr->size = size;
    ptr = memptr;
    ptr += sizeof(size_t);
    ptr += sizeof(MemInfo *);
    ptr += sizeof(size_t *);
    ptr += sizeof(size_t *);
    ptr += sizeof(void *);
    memptr->headbufptr = (size_t *) ptr;
    ptr += FERMEM_BUFSIZE * sizeof(size_t);
    memptr->memory = ptr;
    ptr += size;
    memptr->tailbufptr = (size_t *) ptr;
    for (k = 0; k < FERMEM_BUFSIZE; k++)
        memptr->headbufptr[k] = FERMEM_BUFVALUE;
    for (k = 0; k < FERMEM_BUFSIZE; k++)
        memptr->tailbufptr[k] = FERMEM_BUFVALUE;
    /* Return the pointer to part of the memory to be used */
    return memptr->memory;
}

/*
 * Removes the memory list entry associated with the given user memory pointer
 * and returns the pointer to the original allocated memory.
 * The size of the user memory that was allocated is returned as the value 
 * pointed to by sizeptr.
 */
void *RemoveFromMemInfoList(void *ptr, size_t *sizeptr)
{
    MemInfo *prevptr;
    MemInfo *memptr;
    int      k;

    for (prevptr = NULL, memptr = MemInfoList; memptr != NULL; prevptr = memptr, memptr = memptr->next) {
        /* First check the buffers are not corrupt */
        for (k = 0; k < FERMEM_BUFSIZE; k++) {
            if ( memptr->headbufptr[k] != FERMEM_BUFVALUE ) {
                fprintf(stderr, "Memory underwrite detected for allocation %016p of %ld bytes\n", memptr->memory, memptr->size);
                abort();
            }
        }
        for (k = 0; k < FERMEM_BUFSIZE; k++) {
            if ( memptr->tailbufptr[k] != FERMEM_BUFVALUE ) {
                fprintf(stderr, "Memory overwrite detected for allocation %016p of %ld bytes\n", memptr->memory, memptr->size);
                abort();
            }
        }
        if ( memptr->memory == ptr ) {
            /* remove this MemInfo from the list */
            if ( prevptr != NULL )
                prevptr->next = memptr->next;
            else
                MemInfoList = memptr->next;
            memptr->next = NULL;
            /* assign the size of the user memory and return the pointer to the original memory */
            *sizeptr = memptr->size;
            return memptr;
        }
    }
    fprintf(stderr, "Attempt to free unallocated memory %016p\n", memptr);
    abort();
}

#endif

/*
 * Allocates memory like malloc.  If (and only if) the compile flag MEMORYDEBUG is defined, 
 * prints a line to memorydebug.txt with the allocation information.  The value of filename 
 * should be __FILE__ and the value of linenumber should be __LINE__ in the source file 
 * calling this routine.
 *
 * So: 
 *     myptr = malloc(mysize);
 * should be turned into: 
 *     myptr = FerMem_Malloc(mysize, __FILE__, __LINE__);
 */
void *FerMem_Malloc(size_t size, char *filename, int linenumber) 
{
    void *ptr;

#ifdef MEMORYDEBUG
    char msg[256];

    /* allocate required memory as well as for the surrounding MemInfo block */
    ptr = PyMem_Malloc(size + sizeof(MemInfo));
    ptr = AddToMemInfoList(ptr, size);
    /* initialize to non-zero junk to catch uninitialized memory usage */
    memset(ptr, 0x6B, size);
    sprintf(msg, "memory malloc allocated for %ld bytes : file %s : line %d", size, filename, linenumber);
    FerMem_WriteDebugMessage(ptr, ptr + size, msg);

#else

    ptr = PyMem_Malloc(size);

#endif

    return ptr;
}

/*
 * Reallocates memory like realloc.  If (and only if) the compile flag MEMORYDEBUG is defined, 
 * prints a line to memorydebug.txt with the allocation information.  The value of filename 
 * should be __FILE__ and the value of linenumber should be __LINE__ in the source file 
 * calling this routine.
 *
 * So: 
 *     newptr = realloc(myptr, newsize);
 * should be turned into: 
 *     newptr = FerMem_Realloc(myptr, newsize, __FILE__, __LINE__);
 */
void *FerMem_Realloc(void *ptr, size_t size, char *filename, int linenumber) 
{
    void *newptr;

#ifdef MEMORYDEBUG
    void  *origptr;
    size_t oldsize;
    char   msg[256];

    origptr = RemoveFromMemInfoList(ptr, &oldsize);
    sprintf(msg, "memory to be realloc freed for %ld bytes : file %s : line %d", oldsize, filename, linenumber);
    FerMem_WriteDebugMessage(ptr, ptr + oldsize, msg);

    newptr = PyMem_Realloc(origptr, size + sizeof(MemInfo));

    newptr = AddToMemInfoList(newptr, size);
    /* initialize new memory to non-zero junk to catch uninitialized memory usage */
    if ( size > oldsize )
       memset(newptr + oldsize, 0x6B, size - oldsize);
    sprintf(msg, "memory realloc allocated for %ld bytes : file %s : line %d", size, filename, linenumber);
    FerMem_WriteDebugMessage(newptr, newptr + size, msg);

#else

    newptr = PyMem_Realloc(ptr, size);

#endif

    return newptr;
}

/*
 * Frees memory like free.  If (and only if) the compile flag MEMORYDEBUG is defined, 
 * prints a line to memorydebug.txt with the allocation information.  The value of filename 
 * should be __FILE__ and the value of linenumber should be __LINE__ in the source file 
 * calling this routine.
 *
 * So: 
 *     free(mymem);
 * should be turned into: 
 *     FerMem_Free(mymem, __FILE__, __LINE__);
 */
void FerMem_Free(void *ptr, char *filename, int linenumber) 
{
#ifdef MEMORYDEBUG
    void  *origptr;
    size_t size;
    char   msg[256];

    origptr = RemoveFromMemInfoList(ptr, &size);
    sprintf(msg, "memory to be free freed for %ld bytes : file %s : line %d", size, filename, linenumber);
    FerMem_WriteDebugMessage(ptr, ptr + size, msg);

    PyMem_Free(origptr);

#else

    PyMem_Free(ptr);

#endif
}

