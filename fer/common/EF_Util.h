/* EF_Util.h
 *
 * Jonathan Callahan
 * July 10th 1997
 *
 * This is the header file to be included by routines which
 * are part of the Ferret External Function library.
 *
 * 990422 *jc* Changed "work_array_len" to "work_array_lo/hi"
*  V6.0 *acm*  5/06 string results for external functions
*  V6.2 *acm* 11/08 New element of the external function structure, alt_fcn_name
*                  to store the name of a function to call if the arguments are of
*                  a different type than defined in the current function. E.g.
*                  this lets the user reference XCAT with string arguments and
*                  Ferret will run XCAT_STR
* V664  9/10 *kms* Add EF_PYTHON
*       3/12 *kms* Add E and F dimensions; use NFERDIMS
*/


#ifndef _EF_UTIL_H
#define _EF_UTIL_H

/* .................... Includes .................... */

/* DFTYPE, NFERDIMS, and FORTRAN defined in ferret.h */
#include "ferret.h"

/* .................... Defines ..................... */

#define TRUE  1
#define FALSE 0
#define YES   1
#define NO    0

#define LO    0
#define HI    1

#define ATOM_NOT_FOUND 0   /* This should match the atom_not_found parameter in ferret.parm. */
#define FERR_OK 3          /* This should match the ferr_ok parameter in errmsg.parm. */
#define FERR_EF_ERROR 437  /* This should match the ferr_ef_error parameter in errmsg.parm. */

#define EF_VERSION 1.4

#define EF_MAX_NAME_LENGTH 40
#define EF_MAX_DESCRIPTION_LENGTH 128
#define EF_MAX_ARGS 9
#define EF_MAX_WORK_ARRAYS 9
#define EF_MAX_COMPUTE_ARGS 19 /* = EF_MAX_ARGS + EF_MAX_WORK_ARRAYS + 1 */

enum { EF_C=1, EF_F, EF_PYTHON } EF_LANGUAGE_type;

enum { X_AXIS=0, Y_AXIS, Z_AXIS, T_AXIS, E_AXIS, F_AXIS } EF_AXIS_type;

/* The next two lines of parameters need to match numbers in ferret.parm */
enum { CUSTOM=101, IMPLIED_BY_ARGS, NORMAL, ABSTRACT } EF_AXIS_SOURCE_type;
enum { RETAINED=201, REDUCED } EF_AXIS_REDUCTION_type;

/* These parameters need to match numbers in grid_chg_fcns.parm */
enum { FLOAT_ARG=1, STRING_ARG } EF_ARG_type;

/* These parameters need to match numbers in grid_chg_fcns.parm */
enum { FLOAT_RETURN=1, STRING_RETURN } EF_RETURN_type;

enum { CANNOT_ALLOCATE, INSUFFICIENT_DATA } EF_ERROR_type;

/* .................... Typedefs .................... */

typedef struct {
  int  will_be, modulo;
  int  ss_lo, ss_hi, ss_incr;
  double ww_lo, ww_hi, ww_del;
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
  char alt_fcn_name[EF_MAX_NAME_LENGTH];
  int  language;
  int  num_reqd_args, has_vari_args;
  int  num_work_arrays;
  int  work_array_lo[EF_MAX_WORK_ARRAYS][NFERDIMS];
  int  work_array_hi[EF_MAX_WORK_ARRAYS][NFERDIMS];
  int  axis_will_be[NFERDIMS];
  int  axis_reduction[NFERDIMS];
  int  piecemeal_ok[NFERDIMS];
  int  return_type;
  Axis axis[NFERDIMS];

  /* Information specific to each argument of the function */
  int  axis_implied_from[EF_MAX_ARGS][NFERDIMS];
  int  axis_extend_lo[EF_MAX_ARGS][NFERDIMS];
  int  axis_extend_hi[EF_MAX_ARGS][NFERDIMS];
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

/* ................ Global Variables ................ */
/*
 * The memory_ptr, mr_list_ptr and cx_list_ptr are obtained from Ferret
 * and cached whenever they are passed into one of the "efcn_" functions.
 * These pointers can be accessed by the utility functions in libef_util
 * or lib_ef_c_util.  This way the EF writer does not need to see them.
 */
extern DFTYPE *GLOBAL_memory_ptr;
extern int    *GLOBAL_mr_list_ptr;
extern int    *GLOBAL_cx_list_ptr;
extern int    *GLOBAL_mres_ptr;
extern DFTYPE *GLOBAL_bad_flag_ptr;


/* prototypes of external function used in ef funtions and PyFerret C functions */
extern ExternalFunction *ef_ptr_from_id_ptr(int *id_ptr);
void FORTRAN(ef_get_arg_subscripts_6d)(int *id, int steplo[][6], int stephi[][6], int incr[][6]);
void FORTRAN(ef_get_coordinates)(int *id, int *arg, int *axis, int *lo, int *hi, double coords[]);
void FORTRAN(ef_get_box_size)(int *id, int *arg, int *axis, int *lo, int *hi, double sizes[]);
void FORTRAN(ef_get_box_limits)(int *id, int *arg, int *axis, int *lo, int *hi, 
                                double lo_lims[], double hi_lims[]);
extern void FORTRAN(ef_get_one_val_sub)(int *id_ptr, DFTYPE *mem_ptr,
                                        int *arg_ptr, DFTYPE *val_ptr);

/* these are called by the 4D function definitions */
extern void FORTRAN(ef_set_axis_influence_6d)(int *id_ptr, int *arg,
                                              int *xax, int *yax, int *zax,
                                              int *tax, int *eax, int *fax);
extern void FORTRAN(ef_set_axis_inheritance_6d)(int *id_ptr,
                                                int *xax, int *yax, int *zax,
                                                int *tax, int *eax, int *fax);
extern void FORTRAN(ef_set_axis_reduction_6d)(int *id_ptr,
                                              int *xax, int *yax, int *zax,
                                              int *tax, int *eax, int *fax);
extern void FORTRAN(ef_set_piecemeal_ok_6d)(int *id_ptr,
                                            int *xax, int *yax, int *zax,
                                            int *tax, int *eax, int *fax);
extern void FORTRAN(ef_set_work_array_dims_6d)(int *id_ptr, int *iarray,
                                               int *xlo, int *ylo, int *zlo,
                                               int *tlo, int *elo, int *flo,
                                               int *xhi, int *yhi, int *zhi,
                                               int *thi, int *ehi, int *fhi);
extern void FORTRAN(ef_set_work_array_lens_6d)(int *id_ptr, int *iarray,
                                               int *xlen, int *ylen, int *zlen,
                                               int *tlen, int *elen, int *flen);

#endif /* _EF_UTIL_H */

