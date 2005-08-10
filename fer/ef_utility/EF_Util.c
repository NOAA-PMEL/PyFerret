/* EF_Util.c
 *
 * Jonathan Callahan
 * Feb 16th 1998

 * Revisions:
 * for V530 of Ferret - Oct, 2000 *sh* -- make ef_get_mr_list more robust
 *      so it can be called during custom axis creation

 * This file contains all the utility functions which
 * External Functions need in order to interact with
 * EF "objects".
 */


/* .................... Includes .................... */

#include <stdlib.h> 		/* for convenience */
#include <stdio.h>	 	/* for convenience */
#include <string.h> 		/* for convenience */

#include <sys/types.h> 	        /* required for "NULL" */
#include <setjmp.h>             /* required for jmp_buf */

#include "EF_Util.h"
#include "list.h"       /* locally added list library */


/* ................ Global Variables ................ */
/*
 * The memory_ptr, mr_list_ptr and cx_list_ptr are obtained from Ferret
 * and cached whenever they are passed into one of the "efcn_" functions.
 * These pointers can be accessed by the utility functions in libef_util
 * or lib_ef_c_util.  This way the EF writer does not need to see them.
 */
extern float *GLOBAL_memory_ptr;
extern int   *GLOBAL_mr_list_ptr;
extern int   *GLOBAL_cx_list_ptr;
extern int   *GLOBAL_mres_ptr;
extern float *GLOBAL_bad_flag_ptr;


/* ............. Function Declarations .............. */

/* ... Functions called from the EF code .... */

void ef_version_test_( float * );

void ef_set_num_args_( int *, int * );
void ef_set_num_work_arrays_( int *, int * );
void ef_set_work_array_lens_( int *, int *, int *, int *, int *, int * );
void ef_set_work_array_dims_( int *, int *, int *, int *, int *, int *, int *, int *, int *, int * );
void ef_set_has_vari_args_( int *, int * );
void ef_set_axis_inheritance_( int *, int *, int *, int *, int * );
void ef_set_piecemeal_ok_( int *, int *, int *, int *, int * );

void ef_set_axis_influence_( int *, int *, int *, int *, int *, int * );
void ef_set_axis_reduction_( int *, int *, int *, int *, int * );
void ef_set_axis_extend_( int *, int *, int *, int *, int * );

void ef_set_arg_type_( int *, int *, int *);
void ef_set_return_type_( int *, int *);

void ef_get_bad_flags_(int *, float *, float *);
void ef_get_one_val_(int *, int *, float *);

void ef_get_cx_list_(int *);
void ef_get_mr_list_(int *);
void ef_get_mres_(int *);


/* ... Functions called internally .... */

extern ExternalFunction *ef_ptr_from_id_ptr(int *);

void ef_get_one_val_sub_(int *, float *, int *, float *);


/* ............. Function Definitions .............. */


/*
 * Test the EF version number of the Fortran EF against
 * the EF version number of the Ferret code.
 */
void ef_version_test_(float *version)
{
  int int_version=0, ext_version=0;

  int_version = (int) EF_VERSION * 100;
  ext_version = (int) *version * 100;

  if ( ext_version != int_version ) {
	fprintf(stderr, "\n\
ERROR version mismatch:\n\
\tExternal version [%4.2f] does not match \n\
\tFerret version   [%4.2f].\n\
\tPlease upgrade either Ferret or the\n\
\tExternal Function support files from\n\n\
\t\thttp://tmap.pmel.noaa.gov/Ferret/\n\n", *version, EF_VERSION);
  }

  return;
}


/*
 * Set the number of args for a function.
 */
void ef_set_num_args_(int *id_ptr, int *num_args)
{
  ExternalFunction *ef_ptr=NULL;

  if ( *num_args > EF_MAX_ARGS ) {
	fprintf(stderr, "ERROR in ef_set_num_args: %d is greater than maximum: %d\n", *num_args, EF_MAX_ARGS);
  }

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->num_reqd_args = *num_args;

  return;
}


/*
 * Set the number of work arrays requested by a function.
 */
void ef_set_num_work_arrays_(int *id_ptr, int *num_arrays)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->num_work_arrays = *num_arrays;

  return;
}


/*
 * Set the requested size (in words) for a specific work array.
 */
void ef_set_work_array_lens_(int *id_ptr, int *iarray, int *xlen,
	int *ylen, int *zlen, int *tlen)
{
  ExternalFunction *ef_ptr=NULL;
  int array_id = *iarray - 1;      /* F to C conversion */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->work_array_lo[array_id][0] = 1;
  ef_ptr->internals_ptr->work_array_lo[array_id][1] = 1;
  ef_ptr->internals_ptr->work_array_lo[array_id][2] = 1;
  ef_ptr->internals_ptr->work_array_lo[array_id][3] = 1;

  ef_ptr->internals_ptr->work_array_hi[array_id][0] = *xlen;
  ef_ptr->internals_ptr->work_array_hi[array_id][1] = *ylen;
  ef_ptr->internals_ptr->work_array_hi[array_id][2] = *zlen;
  ef_ptr->internals_ptr->work_array_hi[array_id][3] = *tlen;

  return;
}


/*
 * Set the requested lo and hi dimensions for a specific work array.
 */
void ef_set_work_array_dims_(int *id_ptr, int *iarray, 
    int *xlo, int *ylo, int *zlo, int *tlo,
    int *xhi, int *yhi, int *zhi, int *thi)
{
  ExternalFunction *ef_ptr=NULL;
  int array_id = *iarray - 1;      /* F to C conversion */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->work_array_lo[array_id][0] = *xlo;
  ef_ptr->internals_ptr->work_array_lo[array_id][1] = *ylo;
  ef_ptr->internals_ptr->work_array_lo[array_id][2] = *zlo;
  ef_ptr->internals_ptr->work_array_lo[array_id][3] = *tlo;

  ef_ptr->internals_ptr->work_array_hi[array_id][0] = *xhi;
  ef_ptr->internals_ptr->work_array_hi[array_id][1] = *yhi;
  ef_ptr->internals_ptr->work_array_hi[array_id][2] = *zhi;
  ef_ptr->internals_ptr->work_array_hi[array_id][3] = *thi;

  return;
}


/*
 * Set the "variable arguments" flag for a function.
 */
void ef_set_has_vari_args_(int *id_ptr, int *has_vari_args)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->has_vari_args = *has_vari_args;

  return;
}


void ef_set_axis_inheritance_(int *id_ptr, int *ax0, int *ax1, int *ax2, int *ax3)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  if ( *ax0 != CUSTOM && *ax0 != IMPLIED_BY_ARGS && *ax0 != NORMAL && *ax0 != ABSTRACT ) {
	  fprintf(stderr, "ERROR in ef_set_axis_inheritance: Axis type not supported on X axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please check the spelling of the axis inheritance type.\n");
	}
  if ( *ax1 != CUSTOM && *ax1 != IMPLIED_BY_ARGS && *ax1 != NORMAL && *ax1 != ABSTRACT ) {
	  fprintf(stderr, "ERROR in ef_set_axis_inheritance: Axis type not supported on Y axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please check the spelling of the axis inheritance type.\n");
	}
  if ( *ax2 != CUSTOM && *ax2 != IMPLIED_BY_ARGS && *ax2 != NORMAL && *ax2 != ABSTRACT ) {
	  fprintf(stderr, "ERROR in ef_set_axis_inheritance: Axis type not supported on Z axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please check the spelling of the axis inheritance type.\n");
	}
  if ( *ax3 != CUSTOM && *ax3 != IMPLIED_BY_ARGS && *ax3 != NORMAL && *ax3 != ABSTRACT ) {
	  fprintf(stderr, "ERROR in ef_set_axis_inheritance: Axis type not supported on T axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please check the spelling of the axis inheritance type.\n");
	}

  ef_ptr->internals_ptr->axis_will_be[0] = *ax0;
  ef_ptr->internals_ptr->axis_will_be[1] = *ax1;
  ef_ptr->internals_ptr->axis_will_be[2] = *ax2;
  ef_ptr->internals_ptr->axis_will_be[3] = *ax3;

  return;
}


void ef_set_piecemeal_ok_(int *id_ptr, int *ax0, int *ax1, int *ax2, int *ax3)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->piecemeal_ok[0] = *ax0;
  ef_ptr->internals_ptr->piecemeal_ok[1] = *ax1;
  ef_ptr->internals_ptr->piecemeal_ok[2] = *ax2;
  ef_ptr->internals_ptr->piecemeal_ok[3] = *ax3;

  return;
}


void ef_set_axis_limits_(int *id_ptr, int *axis, int *lo, int *hi)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->axis[*axis-1].ss_lo = *lo;
  ef_ptr->internals_ptr->axis[*axis-1].ss_hi = *hi;

  return;
}


void ef_set_axis_influence_(int *id_ptr, int *arg, int *ax0, int *ax1, int *ax2, int *ax3)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  if ( *ax0 != YES && *ax0 != NO ) {
	  fprintf(stderr, "ERROR in ef_set_axis_influence on X axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please use either YES or NO.\n");
	}
  if ( *ax1 != YES && *ax1 != NO ) {
	  fprintf(stderr, "ERROR in ef_set_axis_influence on Y axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please use either YES or NO.\n");
	}
  if ( *ax2 != YES && *ax2 != NO ) {
	  fprintf(stderr, "ERROR in ef_set_axis_influence on Z axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please use either YES or NO.\n");
	}
  if ( *ax3 != YES && *ax3 != NO ) {
	  fprintf(stderr, "ERROR in ef_set_axis_influence on T axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please use either YES or NO.\n");
	}

  ef_ptr->internals_ptr->axis_implied_from[*arg-1][0] = *ax0;
  ef_ptr->internals_ptr->axis_implied_from[*arg-1][1] = *ax1;
  ef_ptr->internals_ptr->axis_implied_from[*arg-1][2] = *ax2;
  ef_ptr->internals_ptr->axis_implied_from[*arg-1][3] = *ax3;

  return;
}


void ef_set_axis_reduction_(int *id_ptr, int *ax0, int *ax1, int *ax2, int *ax3)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  if ( *ax0 != RETAINED && *ax0 != REDUCED ) {
	  fprintf(stderr, "ERROR in ef_set_axis_reduction on X axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please use either RETAINED or REDUCED.\n");
	}
  if ( *ax1 != RETAINED && *ax1 != REDUCED ) {
	  fprintf(stderr, "ERROR in ef_set_axis_reduction on Y axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please use either RETAINED or REDUCED.\n");
	}
  if ( *ax2 != RETAINED && *ax2 != REDUCED ) {
	  fprintf(stderr, "ERROR in ef_set_axis_reduction on Z axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please use either RETAINED or REDUCED.\n");
	}
  if ( *ax3 != RETAINED && *ax3 != REDUCED ) {
	  fprintf(stderr, "ERROR in ef_set_axis_reduction on T axis of %s.\n", ef_ptr->name);
	  fprintf(stderr, "      Please use either RETAINED or REDUCED.\n");
	}

  ef_ptr->internals_ptr->axis_reduction[0] = *ax0;
  ef_ptr->internals_ptr->axis_reduction[1] = *ax1;
  ef_ptr->internals_ptr->axis_reduction[2] = *ax2;
  ef_ptr->internals_ptr->axis_reduction[3] = *ax3;

  return;
}


void ef_set_axis_extend_(int *id_ptr, int *arg, int *axis, int *lo, int *hi)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->axis_extend_lo[*arg-1][*axis-1] = *lo;
  ef_ptr->internals_ptr->axis_extend_hi[*arg-1][*axis-1] = *hi;

  return;
}


void ef_set_arg_type_(int *id_ptr, int *arg, int *arg_type)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->arg_type[*arg-1] = *arg_type;

  return;
}

void ef_set_return_type_(int *id_ptr, int *return_type)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->return_type = *return_type;

  return;
}


void ef_get_one_val_(int *id_ptr, int *arg_ptr, float *val_ptr)
{
  ef_get_one_val_sub_(id_ptr, GLOBAL_memory_ptr, arg_ptr, val_ptr);
}


void ef_get_cx_list_(int *cx_list)
{
  int i=0;

  for (i=0; i<EF_MAX_ARGS; i++) {
    cx_list[i] = GLOBAL_cx_list_ptr[i];
  }
}


void ef_get_mr_list_(int *mr_list)
{
  int i=0;

  if (  GLOBAL_mr_list_ptr != NULL ) {
    for (i=0; i<EF_MAX_ARGS; i++) {
      mr_list[i] = GLOBAL_mr_list_ptr[i];
    }
  } else
    for (i=0; i<EF_MAX_ARGS; i++) {
      mr_list[i] = 0;  /* flag that mr_list isnt available */
    }
}


void ef_get_mres_(int *mres)
{
  *mres = *GLOBAL_mres_ptr;
}




void ef_get_bad_flags_(int *id_ptr, float *bad_flag, float *bad_flag_result)
{
  int i=0;

  for (i=0; i<EF_MAX_ARGS; i++) {
    bad_flag[i] = GLOBAL_bad_flag_ptr[i];
  }
  *bad_flag_result = GLOBAL_bad_flag_ptr[EF_MAX_ARGS];
}


void ef_set_desc_sub_(int *id_ptr, char *text)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(ef_ptr->internals_ptr->description, text);

  return;
}  

void ef_set_arg_desc_sub_(int *id_ptr, int *arg_ptr, char *text)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(ef_ptr->internals_ptr->arg_desc[*arg_ptr-1], text);

  return;
}  

void ef_set_arg_name_sub_(int *id_ptr, int *arg_ptr, char *text)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(ef_ptr->internals_ptr->arg_name[*arg_ptr-1], text);

  return;
}  

void ef_set_arg_unit_sub_(int *id_ptr, int *arg_ptr, char *text)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(ef_ptr->internals_ptr->arg_unit[*arg_ptr-1], text);

  return;
} 

 
void ef_set_custom_axis_sub_(int *id_ptr, int *axis_ptr, float *lo_ptr,
			     float *hi_ptr, float *del_ptr, char *text, int *modulo_ptr)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(ef_ptr->internals_ptr->axis[*axis_ptr-1].unit, text);
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_lo = *lo_ptr;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_hi = *hi_ptr;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_del = *del_ptr;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].modulo = *modulo_ptr;

  return;
}  

 
void ef_set_freq_axis_sub_(int *id_ptr, int *axis_ptr, int *npts, float *box,
			     char *text, int *modulo_ptr)
{

  double lo, hi, del, nfreq, yquist;
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  nfreq = *npts/2;	
  yquist = 0.5* (1./ *box);		/* Nyquist frequency */
  lo = yquist/ nfreq;
  hi = yquist;
  del = lo;

  strcpy(ef_ptr->internals_ptr->axis[*axis_ptr-1].unit, text);
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_lo = lo;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_hi = hi;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_del = del;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].modulo = *modulo_ptr;

  return;
}  
