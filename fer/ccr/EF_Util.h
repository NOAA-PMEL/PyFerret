/* EF_Util.h
 *
 * Jonathan Callahan
 * July 10th 1997
 *
 * This is the header file to be included by routines which
 * are part of the Ferret External Function library.
 *
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

#define EF_VERSION 1.0
#define EF_DELIMITER "--EF--"

#define EF_MAX_NAME_LENGTH 40
#define EF_MAX_DESCRIPTION_LENGTH 128
#define EF_MAX_ARGS 3

enum { EF_C=1, EF_F } EF_LANGUAGE_type;

enum { CONFIGURE=1, COMPUTE } EF_ACTION_type;

enum { X_AXIS=0, Y_AXIS, Z_AXIS, T_AXIS } EF_AXIS_type;

enum { CUSTOM=101, IMPLIED_BY_ARGS, NORMAL, ABSTRACT } EF_AXIS_SOURCE_type;

enum { CANNOT_ALLOCATE, INSUFFICIENT_DATA } EF_ERROR_type;

/* .................... Typedefs .................... */

/*
 * This structure is passed in the DATA_IS_NEXT message
 */
typedef struct {
  int num_datapoints;
  int bytes_per_point;
} DataInfo;

/*
 * This structure is passed in the TELLME_AXIS message
 */
typedef struct {
  int index[2];
  double value[2];
  char unit[EF_MAX_NAME_LENGTH];
  int it_is_regular;
} AxisInfo;

/*
 * This structure is passed in the COMPUTE message
 */
typedef struct {
  int narg, num_datapoints;
  int loss[EF_MAX_ARGS][4];
  int hiss[EF_MAX_ARGS][4];
  float bad_flag[EF_MAX_ARGS+1]; /* +1 for the result */
  float *data[EF_MAX_ARGS+1]; /* +1 for the result */
} ComputeInfo;

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
  int  num_reqd_args, has_vari_args;
  int  axis_will_be[4];
  int  piecemeal_ok[4];

  /* Information specific to each argument of the function */
  int  axis_implied_from[EF_MAX_ARGS][4];
  int  axis_extend_lo[EF_MAX_ARGS][4];
  int  axis_extend_hi[EF_MAX_ARGS][4];
  char arg_name[EF_MAX_ARGS][EF_MAX_NAME_LENGTH];
  char arg_units[EF_MAX_ARGS][EF_MAX_NAME_LENGTH];
  char arg_descr[EF_MAX_ARGS][EF_MAX_DESCRIPTION_LENGTH];
} ExternalFunctionInternals;
 
/*
 * This structure defines the basic element of the 
 * GLOBAL_ExternalFunctionList.
 */
typedef struct {
  void *handle;
  char name[EF_MAX_NAME_LENGTH];
  char path[EF_MAX_DESCRIPTION_LENGTH];
  int id, already_have_internals, language;
  ExternalFunctionInternals *internals_ptr;
} ExternalFunction;


#endif	/* _EF_UTIL_H */

