/*
 *
 * Utility functions for reading binary data
 *
 * $Id$
 *
 *
 * *kob*  4/06 v600 - changes for 64-bit build 
 *
 * * 1/12 *acm* - Ferret 6.8 Changes for double-precision ferret,
 *                see the definition of macro DFTYPE in binaryRead.h
 *   2/12 *kms* - Add E and F dimensions
 */

#ifdef MAC_SSIZE
typedef long ssize_t;
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include "binaryRead.h"

                                /* FORTRAN interface variables */
static FileInfo *FFileInfo = 0; /* Implies only one open file at a time */
static int Permutes[MAXDIMS];   /* Current permute array */
static int Swap = 0;            /* Swap bytes flag */
static char Errbuf[1024];       /* Store error messages */

#define MAXTYPES 256
static struct {
  int length;
  char type[MAXTYPES];
} Types;

static void freeMemory(FileInfo *);


static int checkMem(void *p){
  if (p == (void *)0){
    fprintf(stderr, "Out of memory");
    return 0;
  }
  return 1;
}

static void setError(char *str, char *mess) {
  sprintf(Errbuf, str, mess);
}

static void tidyUp(FileInfo *file) {
  close(file->fd);
  freeMemory(file);
}  

static int errReturn(FileInfo *file) {
  tidyUp(file);
  return 0;
}

static int okReturn(FileInfo *file) {
  tidyUp(file);
  return 1;
}

static char *grabMemChunk(FileInfo *file){
                                /* TODO -- mmap flags vary between Unixes */
  MemInfo *mi = &file->minfo;
  freeMemory(file);

                                /* Align new chunk w/page */
  {
    int numPages = mi->filePos / file->pageSize;
    int extra = mi->filePos % file->pageSize;
    int position = numPages * file->pageSize;
    int chunkSize = file->size - position;
    if (chunkSize > MEM_INFO_BLOCKSIZE){
      chunkSize = MEM_INFO_BLOCKSIZE;
    }
    mi->data = mmap(0, chunkSize, PROT_READ, MAP_SHARED, file->fd, position);
    mi->relPos = extra;
    mi->size = chunkSize;
    mi->fileStartPos = position;
  }
  if (mi->data <= (char *)0){
    mi->data = 0;
    setError("Can't allocate enough memory for file %s", file->name);
  }
  return mi->data;
}

static void *initMemory(FileInfo *file) {
                                /* Set up memory mapping */
  MemInfo *mi = &file->minfo;
  mi->data = 0;
  mi->size = 0;
  mi->relPos = 0;
  mi->filePos = file->skip;        /* Skip over bytes at start of file */
  if (!grabMemChunk(file)){
    return 0;
  }
  return mi->data + mi->relPos;
}

static void *nextMemory(FileInfo *file, int amount) {
  MemInfo *mi = &file->minfo;
  mi->relPos += amount;
  mi->filePos += amount;
  if (mi->fileStartPos + mi->size < file->size){
    if (mi->relPos > mi->size - MEM_INFO_MINTHRESH){
      if (!(grabMemChunk(file)))
        return 0;
    }
  }
  return mi->data + mi->relPos;
}

static void freeMemory(FileInfo *file) {
  MemInfo *mi = &file->minfo;
  if (mi->data > (char *)0){
    munmap(mi->data, mi->size);
  }
}

static FileInfo *createBinaryReader(char *name, int lengths[MAXDIMS],
                             int permutes[MAXDIMS], int skip, int swap){
  FileInfo *fi = (FileInfo *)calloc(1, sizeof(FileInfo));
  int i;
                                /* Open file */
  if (!checkMem(fi)){
    return 0;
  }
  Errbuf[0] = '\0';
  fi->pageSize = getpagesize();
  fi->name = (char *)malloc(strlen(name)+1);
  fi->doSwap = swap;
  if (!checkMem(fi->name)){
    return 0;
  }
  strcpy(fi->name, name);
  fi->vindex = MAXDIMS-1;
  for (i=0; i < MAXDIMS; ++i){
    fi->lengths[i] = lengths[i];
    assert(permutes[i] >= 0 && permutes[i] <= 6);
    fi->permutes[i] = permutes[i];
    if (permutes[i] == MAXDIMS-1){
      fi->vindex = i;
    }
  }

  fi->coeffs[0] = 1;
  for (i = 1; i < MAXDIMS-1; i++)
     fi->coeffs[i] = lengths[i-1] * fi->coeffs[i-1];
  fi->coeffs[MAXDIMS-1] = 0;        /* Dimension for variables */

  fi->skip = skip;
  fi->debug = 0;
  fi->vars = 0;
  fi->nvars = 0;
  fi->size = 0;
  fi->fd = open(fi->name, O_RDONLY);
  if (fi->fd < 0){
    setError("Can't open file %s for reading", name);
    return 0;
  }
  {
    struct stat statbuf;
    if (fstat(fi->fd, &statbuf) < 0){
      setError("Can't get size of file %s", fi->name);
      return 0;
    }
    fi->size = statbuf.st_size;
  }

  return fi;
}

static void deleteVar(VarInfo *theVar) {
  free(theVar);
}

static void deleteBinaryReader(FileInfo *fi){
  free(fi->vars);
  tidyUp(fi);
  free(fi->name);
  free(fi);
}

static int addVar(FileInfo *fi, DFTYPE *data, int type, int doRead){
  VarInfo *theVar = 0;
  int i;

  if (fi->vars == (VarInfo *)0){
    fi->vars = malloc(sizeof(VarInfo));
  } else {
    fi->vars = (VarInfo *)realloc(fi->vars, sizeof(VarInfo)*(fi->nvars+1));
  }
  if (!checkMem(fi->vars)){
    return 0;
  }
  theVar = &fi->vars[fi->nvars];
  ++fi->nvars;
  theVar->data = data;
  theVar->doRead = doRead;
  theVar->type = type;
  switch(type){
  case 'b':
    theVar->dataSize = sizeof(char);
    break;
  case 's':
    theVar->dataSize = sizeof(short);
    break;
  case 'i':
    theVar->dataSize = sizeof(int);
    break;
  case 'f':
    theVar->dataSize = sizeof(float);
    break;
  case 'd':
    theVar->dataSize = sizeof(double);
    break;
  default:
    abort();
  }
  fi->lengths[MAXDIMS-1] = fi->nvars; 
  return 1;
}


static void SWAP(unsigned char *p1, unsigned char *p2)
{
  unsigned char c = *p2;
  *p2 = *p1;
  *p1 = c;
}

/* switch the order of the bytes in a long integer */
static int SWAP32(void *i_in)
{
  unsigned char *inptr = (unsigned char *)i_in;
  SWAP(inptr, &inptr[3]);
  SWAP(&inptr[1], &inptr[2]);
}
 
/* switch the order of the bytes in a short integer */
static void SWAP16(void *i_in)
{
  unsigned char *inptr = (unsigned char *)i_in;
  SWAP(inptr, &inptr[1]);
}
 
static double SWAP64(void *i_in)
{
  unsigned char *inptr = (unsigned char *)i_in;
  SWAP(inptr, &inptr[7]);
  SWAP(&inptr[1], &inptr[6]);
  SWAP(&inptr[2], &inptr[5]);
  SWAP(&inptr[3], &inptr[4]);
}

static int readVars(FileInfo *file) {
  int i,j,k,l,m,n,v;
  DFTYPE *dst;
  char *src = initMemory(file);
  int *permutes = file->permutes;
  int *lengths = file->lengths;
  int *coeffs = file->coeffs;
  int indexes[MAXDIMS];
  union {
    short s;
    int i;
    float f;
    double d;
  } buf;                        /* Union needed to insure correct data align */
  VarInfo *var;
  int dataSize;
  char type;
  int index;

  assert( MAXDIMS == 7 );
  for (v=0; v < lengths[permutes[6]]; ++v){
    indexes[6] = v;
    for (n=0; n < lengths[permutes[5]]; ++n){
      indexes[5] = n;
      for (m=0; m < lengths[permutes[4]]; ++m){
        indexes[4] = m;
        for (l=0; l < lengths[permutes[3]]; ++l){
          indexes[3] = l;
          for (k=0; k < lengths[permutes[2]]; ++k){
            indexes[2] = k;
            for (j=0; j < lengths[permutes[1]]; ++j){
              indexes[1] = j;
              for (i=0; i < lengths[permutes[0]]; ++i){

                if (src == NULL){
                  return 0;
                }
                indexes[0] = i;
                var = &file->vars[indexes[file->vindex]];
                dataSize = var->dataSize;
                if (var->doRead){
                  /* Get the permuted index */
                  index = indexes[0] * coeffs[permutes[0]] +
                          indexes[1] * coeffs[permutes[1]] +
                          indexes[2] * coeffs[permutes[2]] +
                          indexes[3] * coeffs[permutes[3]] +
                          indexes[4] * coeffs[permutes[4]] +
                          indexes[5] * coeffs[permutes[5]] +
                          indexes[6] * coeffs[permutes[6]];
                  dst = var->data + index;
                  type = var->type;
                  switch(type){
                  case 'b':
                    *dst = (DFTYPE) *(char *)src;
                    break;
                  case 's':
                    memcpy(&buf.s, src, sizeof(short));
                    if (file->doSwap)
                      SWAP16(&buf.s);
                    *dst = (DFTYPE) buf.s;
                    break;
                  case 'i':
                    memcpy(&buf.i, src, sizeof(int));
                    if (file->doSwap)
                      SWAP32(&buf.i);
                    *dst = (DFTYPE) buf.i;
                    break;
                  case 'f':
                    memcpy(&buf.f, src, sizeof(float));
                    if (file->doSwap)
                      SWAP32(&buf.f);
                    *dst = (DFTYPE) buf.f;
                    break;
                  case 'd':
                    memcpy(&buf.d, src, sizeof(double));
                    if (file->doSwap)
                      SWAP64(&buf.d);
                    *dst = (DFTYPE) buf.d;
                    break;
                  default:
                    abort();
                  }
#ifdef DEBUG_MEy
                  printf("%f at (%d,%d,%d,%d,%d,%d,%d)\n", *dst, i, j, k, l, m, n, v);
#endif
                }
                src = nextMemory(file, dataSize);

              }
            }
          }
        }
      }
    }
  }
  return 1;
}

static int readBinary(FileInfo *file){
                                /* Screen out zero length files */
  if (file->size == 0){
    setError("File %s is an empty file", file->name);
    return errReturn(file);
  }

                                /* Sanity check for size match */
  {
    int dimLength = 1, totalLength = 0, i;
    for (i=0; i < MAXDIMS-1; ++i){ /* Length of all non variable dimensions */
      dimLength *= file->lengths[i];
    }
    for (i=0; i < file->nvars; ++i){
      totalLength += file->vars[i].dataSize * dimLength;
    }
    if (totalLength > file->size - file->skip){
      setError("Size of file %s doesn't match size specified by variables/grid",
               file->name);
      return errReturn(file);
    }
  }


  if (!readVars(file)){
    return errReturn(file);
  }

  return okReturn(file);
}
  
int FORTRAN(br_open)(char *name, int lengths[MAXDIMS],
                                  int permutes[MAXDIMS], int *iskip){
  int skip = (*iskip) * sizeof(DFTYPE); /* Words -> bytes */
  assert(FFileInfo == 0);
  FFileInfo = createBinaryReader(name, lengths, permutes, skip, Swap);
  return FFileInfo != 0;
}

int FORTRAN(br_add_var)(DFTYPE *data, int *doRead) {
  assert(FFileInfo != 0);
  assert(Types.length > 0);
  if (Types.length != 1 && FFileInfo->nvars >= Types.length){
    setError("%s",
             "Number of args in /type doesn't match number of variables");
    return 0;
  }
  {
    char type;
    if (Types.length == 1){        /* All variables same type */
      type = Types.type[0];
    } else {
      type = Types.type[FFileInfo->nvars];
    }
    return addVar(FFileInfo, data, type, *doRead);
  }
}

int FORTRAN(br_read)() {
  assert(FFileInfo != 0);
  return readBinary(FFileInfo);
}

void FORTRAN(br_close)() {
  if (FFileInfo != 0){
    deleteBinaryReader(FFileInfo);
    FFileInfo = 0;
  }
}

void FORTRAN(br_get_error)(char *buf) {
  strcpy(buf, Errbuf);
}

void FORTRAN(br_get_permutes)(int *permutes) {
  int i;
  for (i=0; i < MAXDIMS; ++i){
    permutes[i] = Permutes[i];
  }
}

void FORTRAN(br_set_atts)(int *permutes, int *swap) {
  int i;
  for (i=0; i < MAXDIMS; ++i){
    Permutes[i] = permutes[i]-1; /* FORTRAN -> C indexing */
  }
  Swap = *swap;
}

static int get_type(char *type, char *result) {
  char c1, c2;
  if (strlen(type) != 2)
    return 0;
  c1 = tolower(*type);
  c2 = tolower(type[1]);
  switch(c1){
  case 'i':
    switch(c2){
    case '1':
      *result = 'b';
      return 1;
    case '2':
      *result = 's';
      return 1;
    case '4':
      *result = 'i';
      return 1;
    }
    return 0;
  case 'r':
    switch(c2){
    case '4':
      *result = 'f';
      return 1;
    case '8':
      *result = 'd';
      return 1;
    }
    return 0;
  }
  return 0;
}

/* Type string can be either one item, which implies that all
 * variables are the same type, or a comma delimited set of
 * types
 */
int FORTRAN(br_set_type)(char *type) {
  char buf[1024];
  char *token, *cp, *cp1;
  char result;
  int count = 0;

                                /* Zero out types */
  Types.length = 0;

                                /* Strip out white space */
  cp = type;
  cp1 = buf;
  while (*cp != '\0'){
    if (*cp != ' ' && *cp != '\t'){
      *cp1++ = *cp;
    }
    ++cp;
  }
  *cp1 = '\0';

                                /* Get comma delimited list */
  token = strtok(buf, ",");
  while (token != 0 && *token != '\0'){
    if (!get_type(token, &result)){
      setError("Bad argument to /type -- %s", token);
      return 0;
    }
    ++Types.length;
    Types.type[count++] = result;
    token = strtok(0, ",");
  }
  return 1;
}

