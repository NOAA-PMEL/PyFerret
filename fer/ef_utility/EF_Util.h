/* EF_Util.h
 *
 * Jonathan Callahan
 * July 10th 1997
 *
 * This is the header file to be included by routines which
 * are part of the Ferret External Function library.
 *
 * 990422 *jc* Changed "work_array_len" to "work_array_lo/hi"
 */
 
 
#ifndef	_EF_UTIL_H
#define	_EF_UTIL_H
 
/* .................... Includes .................... */
 


/* .................... Defines ..................... */

#define TRUE  1
#define FALSE 0
#define YES   1
#define NO    0

#define LO    0
#define HI    1

#define ATOM_NOT_FOUND 0  /* This should match the atom_not_found parameter in ferret.parm. */
#define FERR_OK 3  /* This should match the ferr_ok parameter in errmsg.parm. */
#define FERR_EF_ERROR 437  /* This should match the ferr_ef_error parameter in errmsg.parm. */

#define EF_VERSION 1.3

#define EF_MAX_NAME_LENGTH 40
#define EF_MAX_DESCRIPTION_LENGTH 128
#define EF_MAX_ARGS 9
#define EF_MAX_WORK_ARRAYS 9
#define EF_MAX_COMPUTE_ARGS 19 /* = EF_MAX_ARGS + EF_MAX_WORK_ARRAYS + 1 */

enum { EF_C=1, EF_F } EF_LANGUAGE_type;

enum { X_AXIS=0, Y_AXIS, Z_AXIS, T_AXIS } EF_AXIS_type;

/* The next two lines of parameters need to match numbers in ferret.parm */
enum { CUSTOM=101, IMPLIED_BY_ARGS, NORMAL, ABSTRACT } EF_AXIS_SOURCE_type;
enum { RETAINED=201, REDUCED } EF_AXIS_REDUCTION_type;

/* These parameters need to match numbers in grid_chg_fcns.parm */
enum { FLOAT_ARG=1, STRING_ARG } EF_ARG_type;

enum { CANNOT_ALLOCATE, INSUFFICIENT_DATA } EF_ERROR_type;

/* .................... Typedefs .................... */

typedef struct {
  int  will_be, modulo;
  int  ss_lo, ss_hi, ss_incr;
  float ww_lo, ww_hi, ww_del;
  char unit[EF_MAX_NAME_LENGTH];
} Axis;


/*
 * This structure defines the information we can know about
 * an internal function.  Ferret gets information from this
 * structure by calling one of the C routines beginning with
 * "efcn_".
 */
typedef struct {
  /* Information about the overall function */
  float version;
  char description[EF_MAX_DESCRIPTION_LENGTH];
  int  language;
  int  num_reqd_args, has_vari_args;
  int  num_work_arrays;
  int  work_array_lo[EF_MAX_WORK_ARRAYS][4];
  int  work_array_hi[EF_MAX_WORK_ARRAYS][4];
  int  axis_will_be[4];
  int  axis_reduction[4];
  int  piecemeal_ok[4];
  Axis axis[4];

  /* Information specific to each argument of the function */
  int  axis_implied_from[EF_MAX_ARGS][4];
  int  axis_extend_lo[EF_MAX_ARGS][4];
  int  axis_extend_hi[EF_MAX_ARGS][4];
  int  arg_type[EF_MAX_ARGS];
  char arg_name[EF_MAX_ARGS][EF_MAX_NAME_LENGTH];
  char arg_unit[EF_MAX_ARGS][EF_MAX_NAME_LENGTH];
  char arg_desc[EF_MAX_ARGS][EF_MAX_DESCRIPTION_LENGTH];
} ExternalFunctionInternals;
 
/*
 * This structure defines the basic element of the 
 * GLOBAL_ExternalFunctionList.
 */
typedef struct {
  void *handle;
  char name[EF_MAX_NAME_LENGTH];
  char path[EF_MAX_DESCRIPTION_LENGTH];
  int id, already_have_internals;
  ExternalFunctionInternals *internals_ptr;
} ExternalFunction;


#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif

#endif	/* _EF_UTIL_H */

