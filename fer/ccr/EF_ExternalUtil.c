/* EF_ExternalUtil.c
 *
 * Jonathan Callahan
 * Feb 16th 1998
 *
 * This file contains all the utility functions which
 * External Functions need in order to interact with
 * EF "objects".
 */


/* .................... Includes .................... */

#include <stdio.h>	 	/* for convenience */
#include <stdlib.h> 		/* for convenience */
#include <string.h> 		/* for convenience */

#include <sys/types.h> 	        /* required for "NULL" */

#include "EF_Util.h"
#include "/home/r3/tmap/local/sun/include/list.h"  /* locally added list library */


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

void ef_set_num_args_( int *, int * );
void ef_set_has_vari_args_( int *, int * );
void ef_set_axis_inheritance_( int *, int *, int *, int *, int * );
void ef_set_piecemeal_ok_( int *, int *, int *, int *, int * );

void ef_set_axis_influence_( int *, int *, int *, int *, int *, int * );
void ef_set_axis_extend_( int *, int *, int *, int *, int * );

void ef_get_res_subscripts_(int *, int *, int *, int *);
void ef_get_arg_subscripts_(int *, int *, int *, int *);
void ef_get_arg_ss_extremes_(int *, int *, int *);
void ef_get_one_val_(int *, int *, float *);
void ef_get_bad_flags_(int *, float *, float *);

void ef_set_desc_sub_(int *, char *);

void ef_get_coordinates_(int *, int *, int *, int *, int *, float *);
void ef_get_box_size_(int *, int *, int *, int *, int *, float *);

void ef_get_hidden_variables_(int *, int *);


/* ... Functions called internally .... */

ExternalFunction *ef_ptr_from_id_ptr(int *);

int  EF_ListTraverse_FoundID( char *, char * );


void ef_get_res_subscripts_sub_(int *, int *, int *, int *);
void ef_get_arg_subscripts_sub_(int *, int *, int *, int *);
void ef_get_arg_ss_extremes_sub_(int *, int *, int *, int *);
void ef_get_coordinates_sub_(int *, int *, int *, int *, int *, float *);
void ef_get_box_size_sub_(int *, int *, int *, int *, int *, float *);


/* ............. Function Definitions .............. */


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


void ef_set_axis_extend_(int *id_ptr, int *arg, int *axis, int *lo, int *hi)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->axis_extend_lo[*arg-1][*axis-1] = *lo;
  ef_ptr->internals_ptr->axis_extend_hi[*arg-1][*axis-1] = *hi;

  return;
}


void ef_get_res_subscripts_(int *id_ptr, int *res_lo_ss, int *res_hi_ss, int *res_incr)
{
  ef_get_res_subscripts_sub_(GLOBAL_mres_ptr, res_lo_ss, res_hi_ss, res_incr);
}


void ef_get_arg_subscripts_(int *id_ptr, int *arg_lo_ss, int *arg_hi_ss, int *arg_incr)
{
  ef_get_arg_subscripts_sub_(GLOBAL_cx_list_ptr, arg_lo_ss, arg_hi_ss, arg_incr);
}


void ef_get_arg_ss_extremes_(int *id_ptr, int *ss_min, int *ss_max)
{
  ExternalFunction *ef_ptr=NULL;
  int num_args=0;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  num_args = ef_ptr->internals_ptr->num_reqd_args;

  ef_get_arg_ss_extremes_sub_(GLOBAL_cx_list_ptr, &num_args, ss_min, ss_max);
}


void ef_get_one_val_(int *id_ptr, int *arg_ptr, float *val_ptr)
{
  ef_get_one_val_sub_(arg_ptr, GLOBAL_memory_ptr, GLOBAL_mr_list_ptr, GLOBAL_cx_list_ptr, val_ptr);
}


void ef_get_coordinates_(int *id_ptr, int *arg_ptr, int *dim_ptr, int *lo_lim_ptr,
			 int *hi_lim_ptr, float *val_ptr)
{
  ef_get_coordinates_sub_(GLOBAL_cx_list_ptr, arg_ptr, dim_ptr, lo_lim_ptr, hi_lim_ptr, val_ptr);
}


void ef_get_box_size_(int *id_ptr, int *arg_ptr, int *dim_ptr, int *lo_lim_ptr,
			 int *hi_lim_ptr, float *val_ptr)
{
  ef_get_box_size_sub_(GLOBAL_cx_list_ptr, arg_ptr, dim_ptr, lo_lim_ptr, hi_lim_ptr, val_ptr);
}


void ef_get_hidden_variables_(int *cx_list, int *mres)
{
  int i=0;

  for (i=0; i<EF_MAX_ARGS; i++) {
    cx_list[i] = GLOBAL_cx_list_ptr[i];
  }
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
			     float *hi_ptr, float* del_ptr, char *text, int *modulo_ptr)
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
