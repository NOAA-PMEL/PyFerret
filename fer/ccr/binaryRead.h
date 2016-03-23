#ifndef BINARYREAD_H
#define BINARYREAD_H
/*
 * Utility functions for reading binary data
 *
 * *acm*  5/07 v603 - fix prototype of createBinaryRead to use MAXDIMS rather 
 *                    than hardwired to 4; to match what is in .c file.
 *			  (Found by Andy Jacobson doing the MAC build.)
 *
 * * 1/12 *acm* - Ferret 6.8 Changes for double-precision ferret,
 *                see the define macro DFTYPE in binaryRead.h
 *   2/12 *kms* - Add E and F dimensions
 */

#define MEM_INFO_BLOCKSIZE      1048576	/* Max mem chunk size */
#define MEM_INFO_MINTHRESH      1024 /* No closer to mmap boundary than this! */

/* Easier way of handling single/double floating-point declarations */
#ifdef double_p
#define DFTYPE double
#else
#define DFTYPE float
#endif


typedef struct _MemInfo {
  char *data;			/* Memory mapped file contents */
  int relPos;			/* Position relative to mem block start */
  int filePos;			/* Position relative to file start */
  int fileStartPos;		/* Position of memory chunk relative to file origin */
  int size;			/* Size of current memory block */
} MemInfo;

typedef struct _VarInfo {
  /*  Passed values */
  char type;			/* Data type 'b', 's', 'i', 'f', 'd' */
  int doRead;			/* If true, read data */
  DFTYPE *data;			/* Data for variable -- assumed preallocated */
  
  /* Calculated values */
  int dataSize;			/* Size of variable data type */
} VarInfo;

/* One additional dimension for variables */
#define MAXDIMS  7

typedef struct _FileInfo {
  MemInfo minfo;		/* Memory mapped file stuff */
  char *name;			/* Name of file containing binary data */
  int skip;			/* Number of bytes to skip at start of file */
  int debug;			/* Debug flag */
  VarInfo *vars;		/* List of variables to read */
  int nvars;			/* Number of variables */
  int fd;			/* File handle */
  int lengths[MAXDIMS];		/* Lengths of x,y,z,t */
  int coeffs[MAXDIMS];		/* Coefficients calc. from permute/length */
  int permutes[MAXDIMS];	/* ijkl permutations from file -> memory */
  int vindex;			/* Permuted index that is the variable index */
  int size;			/* Size of file in bytes */
  int pageSize;			/* System pagesize */
  int doSwap;			/* Swap bytes */
} FileInfo;

#ifndef FORTRAN
#define FORTRAN(a) a##_
#endif

#endif
