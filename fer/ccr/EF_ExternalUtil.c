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
extern float *GLOBAL_bad_flag_ptr;

/* ............. Function Declarations .............. */

/* ... Functions called from the EF code .... */

void ef_set_num_args_( int *, int * );
void ef_set_num_args( int *, int );
void ef_set_has_vari_args_( int *, int * );
void ef_set_axis_inheritance_( int *, int *, int *, int *, int * );
void ef_set_piecemeal_ok_( int *, int *, int *, int *, int * );

void ef_set_axis_influence_( int *, int *, int *, int *, int *, int * );
void ef_set_axis_extend_( int *, int *, int *, int *, int * );

void ef_get_subscripts_(int *, int *, int *, int *);
void ef_get_one_val_(int *, int *, float *);
void ef_get_bad_flags_(int *, float *, float *);

void ef_set_desc_sub_(int *, char *);

/* ... Functions called internally .... */

ExternalFunction *ef_ptr_from_id_ptr(int *);

int  EF_ListTraverse_FoundID( char *, char * );


void ef_get_subscripts_sub_(int *, int *, int *, int *);


/* ............. Function Definitions .............. */


/*
 * Set the number of args for a function.
 */
void ef_set_num_args_(int *id_ptr, int *num_args)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->num_reqd_args = *num_args;

  return;
}


/*
 * Set the number of args for a function. (C version)
 */
void ef_set_num_args(int *id_ptr, int num_args)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->num_reqd_args = num_args;

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


/*
 * Set the "variable arguments" flag for a function.
 */
void ef_set_axis_inheritance_(int *id_ptr, int *ax0, int *ax1, int *ax2, int *ax3)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->axis_will_be[0] = *ax0;
  ef_ptr->internals_ptr->axis_will_be[1] = *ax1;
  ef_ptr->internals_ptr->axis_will_be[2] = *ax2;
  ef_ptr->internals_ptr->axis_will_be[3] = *ax3;

  return;
}


/*
 * Set the "variable arguments" flag for a function.
 */
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


/*
 * Set the "variable arguments" flag for a function.
 */
void ef_set_axis_influence_(int *id_ptr, int *arg, int *ax0, int *ax1, int *ax2, int *ax3)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->axis_implied_from[*arg-1][0] = *ax0;
  ef_ptr->internals_ptr->axis_implied_from[*arg-1][1] = *ax1;
  ef_ptr->internals_ptr->axis_implied_from[*arg-1][2] = *ax2;
  ef_ptr->internals_ptr->axis_implied_from[*arg-1][3] = *ax3;

  return;
}


/*
 * Set the "variable arguments" flag for a function.
 */
void ef_set_axis_extend_(int *id_ptr, int *arg, int *axis, int *lo, int *hi)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr->internals_ptr->axis_extend_lo[*arg-1][*axis-1] = *lo;
  ef_ptr->internals_ptr->axis_extend_hi[*arg-1][*axis-1] = *hi;

  return;
}


void ef_get_subscripts_(int *id_ptr, int *ss_lo_lim, int *ss_hi_lim, int *ss_incr)
{
  ef_get_subscripts_sub_(GLOBAL_cx_list_ptr, ss_lo_lim, ss_hi_lim, ss_incr);
}


void ef_get_one_val_(int *id_ptr, int *arg_ptr, float *val_ptr)
{
  ef_get_one_val_sub_(arg_ptr, GLOBAL_memory_ptr, GLOBAL_mr_list_ptr, GLOBAL_cx_list_ptr, val_ptr);
}


void ef_get_bad_flags_(int *id_ptr, float *bad_flag, float *bad_flag_result)
{
  int i=0;

  for (i=0; i<EF_MAX_ARGS+1; i++) {
    bad_flag[i] = GLOBAL_bad_flag_ptr[i];
    *bad_flag_result = GLOBAL_bad_flag_ptr[EF_MAX_ARGS];
  }
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
