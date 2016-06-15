#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <netinet/in.h> /* for htonl; header probably varies by platform */
#include <assert.h>

/*
 * if clobber == 0, open filename for overwriting.
 * if clobber == 1, open filename for appending
 * if swap == 0, just write the data as is.
 * if swap == 1, byte-swap the data, then write it.
 *
 * write length to file twice, followed by data, all
 * in network byte order (big-endian)
 *
 * returns 0 for success, non-zero for error
 * (sets Unix errno, see strerror for error messages)
 * Richard Rogers 9/03
 * Ansley Manke 9/03  Make inputs all pointers, send in length of
 *                    filename string; put filename in new ptr
 *                    variable. (As in save_c_string.c)

 * Richard Rogers 10/15/03  fixed potential memory leak, check for
 *                          malloc failure, added some assertions,
 *                          use bulk-write if not byte swapping, use
 *                          strncpy to copy non-NULL terminated filename,
 *                          #include <netinet/in.h> for htonl, added
 *                          write_dods_double_

 * Richard Rogers 10/28/03  added #include<string.h> for strncpy,
 *                          fixed bug in error handling
 *                          NOTE: rr 10/28/03 The length parameter is 
 *                          passed as a signed int, while fwrite() 
 *                          appears to return an unsigned int. That's 
 *                          fine as long as length is less than 2e31.
 */

#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif

int write_dods_(char*filename, int* slen, int *clobber, int *swap, 
                int *length, float *data) 
{
  FILE *f;
  int i, length_n, result = 0;
  char* fileptr;

  union {
    int i;
    float f;
  } hack;

  assert(filename);
  assert(sizeof(int) == 4);  /* should be true for majority of platforms */
  assert(sizeof(float) == 4);

  /* allocate memory and save the filename to null-terminated string */

  if ( fileptr = (char *) malloc(sizeof(char) * (*slen + 1) ) ) {
    strncpy (fileptr, filename, *slen);
    fileptr[*slen] = 0;    /* null-terminate the stored string */
  } else goto cleanup;

  if (*clobber)
    f = fopen (fileptr, "wb");
  else
    f = fopen (fileptr, "ab");
  if (!f) goto cleanup;

  length_n = htonl(*length);
  if (fwrite ((const void *) &length_n, sizeof(int), 1, f) != 1)
    goto cleanup;
  if (fwrite ((const void *) &length_n, sizeof(int), 1, f) != 1)
    goto cleanup;

  if (*swap) {
    for (i=0; i < *length; i+=1) {
      hack.f = data[i];
      hack.i = htonl(hack.i);
      if (fwrite ((const void *) &(hack.f), sizeof(float), 1, f) != 1)
        goto cleanup;
    }
  } else {
    if (fwrite ((const void *) data, sizeof(float), *length, f) != *length)
      goto cleanup;
  }

cleanup:
  if (fileptr) free(fileptr);
  result = errno;
  if (f) {
    if (errno) {             /* preserve original error even if close fails */
      fclose (f);
    } else {               
      if (fclose (f))
	result = errno;      /* return the error from close */
    }
  }
  return result;
}


int write_dods_double_(char*filename, int* slen, int *clobber, int *swap, 
                       int *length, double *data) 
{
  FILE *f;
  int i, length_n, result = 0;
  char* fileptr;

  union {
    int i[2];
    double d;
  } hack;

  assert(filename);
  assert(sizeof(int) == 4);
  assert(sizeof(double) == 8);

  /* allocate memory and save the filename to null-terminated string */

  if ( fileptr = (char *) malloc(sizeof(char) * (*slen + 1) ) ) {
    strncpy (fileptr, filename, *slen);
    fileptr[*slen] = 0;    /* null-terminate the stored string */
  } else goto cleanup;

  if (*clobber)
    f = fopen (fileptr, "wb");
  else
    f = fopen (fileptr, "ab");
  if (!f) goto cleanup;

  length_n = htonl(*length);
  if (fwrite ((const void *) &length_n, sizeof(int), 1, f) != 1)
    goto cleanup;
  if (fwrite ((const void *) &length_n, sizeof(int), 1, f) != 1)
    goto cleanup;

  if (*swap) {
    for (i=0; i < *length; i+=1) {
      int t;
      hack.d = data[i];
      t = hack.i[0];
      hack.i[0] = htonl(hack.i[1]);
      hack.i[1] = htonl(t);
      if (fwrite ((const void *) &(hack.d), sizeof(double), 1, f) != 1)
        goto cleanup;
    }
  } else {
    if (fwrite ((const void *) data, sizeof(double), *length, f) != *length)
      goto cleanup;
  }

cleanup:
  if (fileptr) free(fileptr);
  result = errno;
  if (f) {
    if (errno) {             /* preserve original error even if close fails */
      fclose (f);
    } else {               
      if (fclose (f))
	result = errno;      /* return the error from close */
    }
  }
  return result;
}
