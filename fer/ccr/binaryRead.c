/*
 *
 * Utility functions for reading binary data
 *
 * $Id$
 */


#include "binaryRead.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <malloc.h>
#include <unistd.h>
#include <stdarg.h>
#include <ctype.h>
#include <string.h>

				/* FORTRAN interface variables */
static FileInfo *FFileInfo = 0;	/* Implies only one open file at a time */
static int Permutes[MAXDIMS];	/* Current permute array */
static int Swap = 0;		/* Swap bytes flag */
static char Errbuf[1024];	/* Store error messages */

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

#if 0				/* CYGWIN B20 doesn't support varargs */
static void setError(char *str, ...) {
  va_list ap;
  va_start(ap, str);
  vsprintf(Errbuf, str, ap);
  /*  fprintf(stderr, "%s\n", file->errbuf); */
}
#endif

static void setError(char *str, char *mess) {
  fprintf(stderr, str, mess);
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
  if (mi->data <= 0){
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
  mi->filePos = file->skip;	/* Skip over bytes at start of file */
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
    assert(permutes[i] >= 0 && permutes[i] <= 4);
    fi->permutes[i] = permutes[i];
    if (permutes[i] == MAXDIMS-1){
      fi->vindex = i;
    }
  }

  fi->coeffs[0] = 1;
  fi->coeffs[1] = lengths[0];
  fi->coeffs[2] = lengths[1] * fi->coeffs[1];
  fi->coeffs[3] = lengths[2] * fi->coeffs[2];
  fi->coeffs[4] = 0;		/* Dimension for variables */

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
  int i;
  for (i=0; i < fi->nvars; ++i){
    deleteVar(&fi->vars[i]);
  }
  free(fi->vars);
  tidyUp(fi);
  free(fi->name);
  free(fi);
}

static int addVar(FileInfo *fi, float *data, int type, int doRead){
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
    assert(0);
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
  int i,j,k,l,v;
  float *dst;
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
  } buf;			/* Union needed to insure correct data align */

  for (v=0; v < lengths[permutes[4]]; ++v){
    indexes[4] = v;
    for (l=0; l < lengths[permutes[3]]; ++l){
      indexes[3] = l;
      for (k=0; k < lengths[permutes[2]]; ++k){
	indexes[2] = k;
	for (j=0; j < lengths[permutes[1]]; ++j){
	  indexes[1] = j;
	  for (i=0; i < lengths[permutes[0]]; ++i){
	    VarInfo *var;
	    int dataSize;
	    char type;

	    indexes[0] = i;
	    var = &file->vars[indexes[file->vindex]];
	    dataSize = var->dataSize;
	    if (var->doRead){
	      /* Get the permuted index */
	      int index = indexes[0] * coeffs[permutes[0]] +
		indexes[1] * coeffs[permutes[1]] +
		indexes[2] * coeffs[permutes[2]] +
		indexes[3] * coeffs[permutes[3]] +
		indexes[4] * coeffs[permutes[4]];
	      dst = var->data + index;

	      type = var->type;
	      switch(type){
	      case 'b':
		*dst = (float)*(char *)src;
		break;
	      case 's':
		memcpy(&buf.s, src, sizeof(short));
		if (file->doSwap)
		 SWAP16(&buf.s);
		*dst = (float)buf.s;
		break;
	      case 'i':
		memcpy(&buf.i, src, sizeof(int));
		if (file->doSwap)
		  SWAP32(&buf.i);
		*dst = (float)buf.i;
		break;
	      case 'f':
		memcpy(&buf.f, src, sizeof(float));
		if (file->doSwap)
		  SWAP32(&buf.f);
		*dst = buf.f;
		break;
	      case 'd':
		memcpy(&buf.d, src, sizeof(double));
		if (file->doSwap)
		  SWAP64(&buf.d);
		*dst = (float)buf.d;
		break;
	      default:
		assert(0);
	      }
	      
#if 0	    
	      printf("%f at (%d,%d,%d,%d,%d)\n", *dst, i, j, k, l, v);
#endif
	    }
	    src = nextMemory(file, dataSize);
	  }
	}
      }
    }
  }
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
  int skip = (*iskip) * sizeof(float); /* Words -> bytes */
  assert(FFileInfo == 0);
  FFileInfo = createBinaryReader(name, lengths, permutes, skip, Swap);
  return FFileInfo != 0;
}

int FORTRAN(br_add_var)(float *data, int *doRead) {
  assert(FFileInfo != 0);
  assert(Types.length > 0);
  if (Types.length != 1 && FFileInfo->nvars >= Types.length){
    setError("%s",
	     "Number of args in /type doesn't match number of variables");
    return 0;
  }
  {
    char type;
    if (Types.length == 1){	/* All variables same type */
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
  assert(FFileInfo != 0);
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
/**************************** Test code ****************************/

#ifdef COMPILE_TESTS

#if 1
#define IL 1
#define JL 2
#define KL 3
#define LL 4
#define VL 5
#endif
#if 0
#define IL 22
#define JL 23
#define KL 24
#define LL 25
#define VL 5
#endif

static int lengths[] = {IL, JL, KL, LL, VL};
static int permutes[] = {4, 3, 2, 0, 1}; /* vlkij */
static int permutes1[] = {0, 1, 2, 3, 4}; /* ijklv */
static char dtypes[] = {'b','s','i','f','d'};
static char dtypes1[] = {'d', 'f', 'i', 's', 'b'};

static   char datac[JL][IL][KL][LL];	
static   short datas[JL][IL][KL][LL];
static   int datai[JL][IL][KL][LL];
static   float dataf[JL][IL][KL][LL];
static   double datad[JL][IL][KL][LL];

static void createTestFile() {
  int junk = 0;
  int total = 1;
  int i,j,k,l,v;
  FILE *f;
  assert(f = fopen("binaryTest.dat", "w"));
  for (l=0; l < lengths[3]; ++l){
    for (k=0; k < lengths[2]; ++k){
      for (j=0; j < lengths[1]; ++j){
	for (i=0; i < lengths[0]; ++i){
	  datac[j][i][k][l] = junk;
	  datas[j][i][k][l] = junk;
	  datai[j][i][k][l] = junk;
	  dataf[j][i][k][l] = junk;
	  datad[j][i][k][l] = junk;
	  ++junk;
	  junk %= 128;
	}
      }
    }
  }
  for (i=0; i < MAXDIMS-1; ++i){
    total *= lengths[i];
  }

  for (l=0; l < JL; ++l){
    for (k=0; k < IL; ++k){
      for (j=0; j < KL; ++j){
	for (i=0; i < LL; ++i){
	    fwrite(&datac[l][k][j][i], sizeof(char), 1, f);
	    fwrite(&datas[l][k][j][i], sizeof(short), 1, f);
	    fwrite(&datai[l][k][j][i], sizeof(int), 1, f);
	    fwrite(&dataf[l][k][j][i], sizeof(float), 1, f);
	    fwrite(&datad[l][k][j][i], sizeof(double), 1, f);
	}
      }
    }
  }
  fclose(f);
}

static void createTestFile1() {
  int total = 1;
  int i,j,k,l,v;
  double d;
  float fl;
  int i1;
  short s;
  char c;
  FILE *f;
  int junk = 0;
  assert(f = fopen("binaryTest1.dat", "w"));
  fwrite(&d, sizeof(char), 5, f); /* Write 5 bytes of header junk */
  for (v=0; v < lengths[4]; ++v){
    char type  = dtypes1[v];
    junk = 0;
    for (l=0; l < lengths[3]; ++l){
      for (k=0; k < lengths[2]; ++k){
	for (j=0; j < lengths[1]; ++j){
	  for (i=0; i < lengths[0]; ++i){
	    switch(type){
	    case 'b':
	      c = (char)junk;
	      fwrite(&c, sizeof(char), 1, f);
	      break;
	    case 's':
	      s = (short)junk;
	      fwrite(&s, sizeof(short), 1, f);
	      break;
	    case 'i':
	      i1 = (int)junk;
	      fwrite(&i1, sizeof(int), 1, f);
	      break;
	    case 'f':
	      fl = (float)junk;
	      fwrite(&fl, sizeof(float), 1, f);
	      break;
	    case 'd':
	      d = (double)junk;
	      fwrite(&d, sizeof(double), 1, f);
	      break;
	    default:
	      assert(0);
	    }
	    ++junk;
	    junk %= 128;
	  }
	}
      }
    }
  }
  fclose(f);
}

/* Test 1 -- create zero length file */
static void test1() {
  char *file = mktemp("/tmp/junkXXXXXX");
  FILE *f = fopen(file, "w");
  FileInfo *fi = createBinaryReader(file, lengths, permutes, 0, 0);
  if (!readBinary(fi)){
    printf("Test 1 passed\n");
  } else {
    printf("Test 1 failed\n");
  }
  fclose(f);
  unlink(file);
  deleteBinaryReader(fi);
}
	   
/* Test 2 -- file size doesn't match variable spec */
static void test2() {
  char *file = mktemp("/tmp/junkXXXXXX");
  FILE *f = fopen(file, "w");
  FileInfo *fi;
  fprintf(f, "%d", 1);
  fclose(f);
  fi = createBinaryReader(file, lengths, permutes, 0, 0);

  if (!readBinary(fi)){
    printf("Test 2 passed\n");
  } else {
    printf("Test 2 failed\n");
  }
  unlink(file);
  deleteBinaryReader(fi);
}

/* Test 3 -- binary file w/all types in permute order vlkij */
static void test3() {
  int i, totalLength;
  FileInfo *fi;
  float *data;

  printf("Test 3...\n");
  printf("Creating reader ... \n");
  fi = createBinaryReader("binaryTest.dat", lengths, permutes, 0, 0);
  assert(fi);
  data = (float *)malloc(fi->size * sizeof(float) * sizeof(dtypes));

  totalLength = lengths[0] * lengths[1] * lengths[2] * lengths[3];
  for (i=0; i < sizeof(dtypes); ++i){
    addVar(fi, data, dtypes[i], 1);
    data += totalLength;
  }
  printf("Reading binary... \n");
  assert(readBinary(fi));
  {
    int i, j, total=1;
    for (j=0; j < sizeof(dtypes); ++j){
      VarInfo *vi = &fi->vars[j];
      float *theData = vi->data;
      printf("Checking results for type %c...\n", dtypes[j]);
      for (i=0; i < totalLength; ++i){
	assert(theData[i] == i % 128);
      }
      printf("OK for type %c...\n", dtypes[j]);
    }
  }
  deleteBinaryReader(fi);
  printf("End test 3...\n");
}

/* Test 3 -- binary file w/all types in permute order ijklv */
static void test4() {
  int i, totalLength;
  FileInfo *fi;
  float *data;

  printf("Test 4...\n");
  printf("Creating reader ... \n");
  fi = createBinaryReader("binaryTest1.dat", lengths, permutes1, 5, 0);
  assert(fi);
  data = (float *)malloc(fi->size * sizeof(float) * sizeof(dtypes));

  totalLength = lengths[0] * lengths[1] * lengths[2] * lengths[3];
  for (i=0; i < sizeof(dtypes); ++i){
    addVar(fi, data, dtypes1[i], 1);
    data += totalLength;
  }
  printf("Reading binary... \n");
  assert(readBinary(fi));
  {
    int i, j, total=1;
    for (j=0; j < sizeof(dtypes1); ++j){
      VarInfo *vi = &fi->vars[j];
      float *theData = vi->data;
      printf("Checking results for type %c...\n", dtypes1[j]);
      for (i=0; i < totalLength; ++i){
	assert(theData[i] == i % 128);
      }
      printf("OK for type %c...\n", dtypes1[j]);
    }
  }
  deleteBinaryReader(fi);
  printf("End Test 4...\n");
}

int main(int argc, char *argv[]){
  FileInfo *fi = 0;
  int skipCreate = 0;

  if (argc > 1 && strcmp(argv[1], "-skip") == 0){
    skipCreate = 1;
  }

  if (!skipCreate){
    printf("Creating test file ... \n");
    createTestFile();
    createTestFile1();
  } else {
    printf("Skipping file creation. Be warned\n");
  }

  test1();
  test2();
  test3(); 
  test4();

}
#endif /* Compile tests */
