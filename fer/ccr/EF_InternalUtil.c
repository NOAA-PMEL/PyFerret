/* EF_InternalUtil.c
 *
 * Jonathan Callahan
 * Sep 4th 1997
 *
 * This file contains all the utility functions which Ferret
 * needs in order to communicate with an external function.
 */


/* .................... Includes .................... */
 
#include <stdio.h>		/* for convenience */
#include <stdlib.h>		/* for convenience */
#include <string.h>		/* for convenience */
#include <unistd.h>		/* for convenience */
#include <fcntl.h>		/* for fcntl() */
#include <dlfcn.h>		/* for dynamic linking */

#include <sys/types.h>	        /* required for some of our prototypes */
#include <sys/stat.h>
#include <sys/errno.h>

#include "EF_Util.h"
#include "/home/r3/tmap/local/sun/include/list.h"  /* locally added list library */


/* ................ Global Variables ................ */
/*
 * The memory_ptr, mr_list_ptr and cx_list_ptr are obtained from Ferret
 * and cached whenever they are passed into one of the "efcn_" functions.
 * These pointers can be accessed by the utility functions in libef_util
 * or lib_ef_c_util.  This way the EF writer does not need to see them.
 */

static LIST  *GLOBAL_ExternalFunctionList;
float *GLOBAL_memory_ptr;
int   *GLOBAL_mr_list_ptr;
int   *GLOBAL_cx_list_ptr;
int   *GLOBAL_mres_ptr;
float *GLOBAL_bad_flag_ptr;

static int I_have_scanned_already = FALSE;
static int I_have_warned_already = TRUE; /* Warning turned off Jan '98 */


/* ............. Function Declarations .............. */
/*
 * Note that all routines called directly from Ferret,
 * ie. directly from Fortran, should be all lower case,
 * be of type 'void', pass by reference and should end with 
 * an underscore.
 */


/* .... Functions called by Ferret .... */

int  efcn_scan_( int * );
int  efcn_already_have_internals_( int * );

int  efcn_gather_info_( int * );
void efcn_get_custom_axes_( int *, int * );
void efcn_get_result_limits_( int *, float *, int *, int * );
void efcn_compute_( int *, int *, int *, int *, float *, int *, float * );


void efcn_get_custom_axis_sub_( int *, int *, float *, float *, float *, char *, int * );


int  efcn_get_id_( char * );
int  efcn_match_template_( char * );


void efcn_get_name_( int *, char * );
void efcn_get_version_( int *, float * );
void efcn_get_descr_( int *, char * );
int  efcn_get_num_reqd_args_( int * );
void efcn_get_has_vari_args_( int *, int * );
void efcn_get_axis_will_be_( int *, int * );
void efcn_get_piecemeal_ok_( int *, int * );

void efcn_get_axis_implied_from_( int *, int *, int * );
void efcn_get_axis_extend_lo_( int *, int *, int * );
void efcn_get_axis_extend_hi_( int *, int *, int * );
void efcn_get_axis_limits_( int *, int *, int *, int * );
void efcn_get_arg_name_( int *, int *, char * );
void efcn_get_arg_unit_( int *, int *, char * );
void efcn_get_arg_desc_( int *, int *, char * );


/* ... Functions called internally .... */

void EF_force_linking(int);

void EF_store_globals(float *, int *, int *, int *, float *);

ExternalFunction *ef_ptr_from_id_ptr(int *);

int  EF_ListTraverse_fprintf( char *, char * );
int  EF_ListTraverse_FoundName( char *, char * );
int  EF_ListTraverse_MatchTemplate( char *, char * );
int  EF_ListTraverse_FoundID( char *, char * );

int  EF_New( ExternalFunction * );


/* ... FORTRAN Functions available to External Functions ... */




/* .............. Function Definitions .............. */


/* .... Functions for use by Ferret (to be called from Fortran) .... */

/*
 * Note that all routines called directly from Ferret,
 * ie. directly from Fortran, should be all lower case,
 * should pass by reference and should end with an underscore.
 */

int efcn_scan_( int *gfcn_num_internal )
{
  
  FILE *file_ptr=NULL;
  ExternalFunction ef; 
 
  char file[EF_MAX_NAME_LENGTH]="", *path=NULL;
  char paths[8192]="", cmd[EF_MAX_DESCRIPTION_LENGTH]="";
  int count=0, status=LIST_OK;

  static int return_val=0; /* static because it needs to exist after the return statement */

  /*
   * We need to generate calls to all the functions in EF_ExternalUtil.c
   * in order to have Solaris link these symbols into the final executable.
   */
  EF_force_linking(0);

  if ( I_have_scanned_already ) {
    return_val = list_size(GLOBAL_ExternalFunctionList);
    return return_val;
  }

  if ( (GLOBAL_ExternalFunctionList = list_init()) == NULL ) {
    fprintf(stderr, "ERROR: Unable to initialize GLOBAL_ExternalFunctionList.\n");
    return_val = -1;
    return return_val;
  }

  /*
   * - Get all the paths from the "FER_EXTERNAL_FUNCTIONS" environment variable.
   *
   * - While there is another path:
   *    - get the path;
   *    - create a pipe for the "ls -1" command;
   *    - read stdout and use each file name to create another external function entry;
   *
   */

  if ( !getenv("FER_EXTERNAL_FUNCTIONS") ) {
    if ( !I_have_warned_already ) {
      fprintf(stderr, "\nWARNING: environment variable FER_EXTERNAL_FUNCTIONS not defined.\n\n");
      I_have_warned_already = TRUE;
    }
    return_val = 0;
    return return_val;
  }

  sprintf(paths, "%s", getenv("FER_EXTERNAL_FUNCTIONS"));
    
  path = strtok(paths, " \t");
    
  if (path[strlen(path)-1] != '/')
    strcat(path, "/"); 

  if ( path == NULL ) {
 
    fprintf(stderr, "\nWARNING:No paths were found in the environment variable FER_EXTERNAL_FUNCTIONS\n\n");

    return_val = 0;
    return return_val;
 
  } else {
    
    do {

      sprintf(cmd, "ls -1 %s", path);

      if ( (file_ptr = popen(cmd, "r")) == (FILE *) NULL ) {
	fprintf(stderr, "\nERROR: Cannot open pipe.\n\n");
	return_val = -1;
	return return_val;
      }
 
      while ( fgets(file, EF_MAX_NAME_LENGTH, file_ptr) != NULL ) {

	file[strlen(file)-1] = '\0';   /* chop off the carriage return */
	if ( strstr(file, ".so") != NULL ) {
	  *strstr(file, ".so") = '\0'; /* chop off the ".so" */
	  strcpy(ef.path, path);
	  strcpy(ef.name, file);
	  ef.id = *gfcn_num_internal + ++count; /* pre-increment because F arrays start at 1 */
	  ef.already_have_internals = NO;
	  ef.internals_ptr = NULL;
	  list_insert_after(GLOBAL_ExternalFunctionList, &ef, sizeof(ExternalFunction));
	}
      }
 
      pclose(file_ptr);
 
      path = strtok(NULL, " \t");
 
    } while ( path != NULL );

    I_have_scanned_already = TRUE;
  }

  return_val = count;
  return return_val;

}


/*
 * Determine whether an external function has already 
 * had its internals read.
 */
int efcn_already_have_internals_( int *id_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;

  static int return_val=0; /* static because it needs to exist after the return statement */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return return_val; }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

  return_val = ef_ptr->already_have_internals;

  return return_val;
}


/*
 * Find an external function based on its integer ID and
 * gather information describing the function. 
 *
 * Return values:
 *     -1: error occurred, dynamic linking was unsuccessful
 *      0: success
 */
int efcn_gather_info_( int *id_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  ExternalFunctionInternals *i_ptr=NULL;
  int i=0, j=0;
  char ef_object[1024]="", tempText[EF_MAX_NAME_LENGTH]="", *c;

  static int return_val=0; /* static because it needs to exist after the return statement */

  void *handle;
  void (*f_init_ptr)(int *);

  /*
   * Find the external function.
   */
  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return return_val; }

  /*
   * Get a handle for the shared object.
   */
  strcat(ef_object, ef_ptr->path);
  strcat(ef_object, ef_ptr->name);
  strcat(ef_object, ".so");

  ef_ptr->handle = dlopen(ef_object, RTLD_LAZY);
  
  /*
   * Allocate and default initialize the internal information.
   * If anything went wrong, return the return_val.
   */
  return_val = EF_New(ef_ptr);

  if ( return_val != 0) {
    return return_val;
  }

  /*
   * Call the external function to really initialize the internal information.
   */
  i_ptr = ef_ptr->internals_ptr;

  if ( i_ptr->language == EF_C ) {

    fprintf(stderr, "\nERROR: C is not a supported language for External Functions.\n\n");
    return_val = -1;
    return return_val;

  } else if ( i_ptr->language == EF_F ) {

    /* Information about the overall function */

    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_init_");
    f_init_ptr = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
    if (f_init_ptr == NULL) {
      fprintf(stderr, "ERROR in efcn_gather_info(): %s is not found.\n", tempText);
      fprintf(stderr, "  dlerror: %s\n", dlerror());
      return -1;
    }

    (*f_init_ptr)(id_ptr);

  }
  
  return 0;
}


/*
 * Find an external function based on its integer ID, 
 * Query the function about custom axes. Store the context
 * list information for use by utility functions.
 */
void efcn_get_custom_axes_( int *id_ptr, int *cx_list_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  char tempText[EF_MAX_NAME_LENGTH]="";

  void (*fptr)(int *);

  /*
   * Store the context list globally.
   */
  EF_store_globals(NULL, NULL, cx_list_ptr, NULL, NULL);

  /*
   * Find the external function.
   */
  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  if ( ef_ptr->internals_ptr->language == EF_F ) {

    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_custom_axes_");

    fptr  = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
    (*fptr)( id_ptr );

  } else {

    fprintf(stderr, "\nExternal Functions in C are not supported yet.\n\n");

  }

  return;
}


/*
 * Find an external function based on its integer ID, 
 * Query the function about abstract axes. Pass memory,
 * mr_list and cx_list info into the external function.
 */
void efcn_get_result_limits_( int *id_ptr, float *memory, int *mr_list_ptr, int *cx_list_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  char tempText[EF_MAX_NAME_LENGTH]="";

  void (*fptr)(int *);

  /*
   * Store the memory pointer and various lists globally.
   */
  EF_store_globals(memory, mr_list_ptr, cx_list_ptr, NULL, NULL);

  /*
   * Find the external function.
   */
  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  if ( ef_ptr->internals_ptr->language == EF_F ) {

    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_result_limits_");

    fptr  = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
    (*fptr)( id_ptr);

  } else {

    fprintf(stderr, "\nExternal Functions in C are not supported yet.\n\n");

  }

  return;
}


/*
 * Find an external function based on its integer ID, 
 * pass the necessary information and the data and tell
 * the function to calculate the result.
 */
void efcn_compute_( int *id_ptr, int *narg_ptr, int *cx_list_ptr, int *mres_ptr,
	float *bad_flag_ptr, int *mr_arg_offset_ptr, float *memory )
{
  ExternalFunction *ef_ptr=NULL;
  int xyzt=0, i=0;
  int arg_points[EF_MAX_ARGS];
  char tempText[EF_MAX_NAME_LENGTH]="";

  int (*fptr)(int, void *);
  void (*f1arg)(int *, float *, float *);
  void (*f2arg)(int *, float *, float *, float *);
  void (*f3arg)(int *, float *, float *, float *, float *);
  void (*f4arg)(int *, float *, float *, float *, float *, float *);
  void (*f5arg)(int *, float *, float *, float *, float *, float *, float *);
  void (*f6arg)(int *, float *, float *, float *, float *, float *, float *,
		float *);
  void (*f7arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *);
  void (*f8arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *);
  void (*f9arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *);

  /*
   * Store the memory pointer and various lists globally.
   */
  EF_store_globals(memory, mr_arg_offset_ptr, cx_list_ptr, mres_ptr, bad_flag_ptr);

  /*
   * Find the external function.
   */
  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  if ( ef_ptr->internals_ptr->language == EF_F ) {

    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_compute_");

    switch ( ef_ptr->internals_ptr->num_reqd_args ) {

    case 1:
      f1arg  = (void (*)(int *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f1arg)( id_ptr, memory + mr_arg_offset_ptr[0], memory + mr_arg_offset_ptr[EF_MAX_ARGS] );
      break;

    case 2:
      f2arg  = (void (*)(int *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f2arg)( id_ptr, memory + mr_arg_offset_ptr[0],
		memory + mr_arg_offset_ptr[1], memory + mr_arg_offset_ptr[EF_MAX_ARGS] );
      break;

    case 3:
      f3arg  = (void (*)(int *, float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f3arg)( id_ptr, memory + mr_arg_offset_ptr[0],
		memory + mr_arg_offset_ptr[1], memory + mr_arg_offset_ptr[2], 
		memory + mr_arg_offset_ptr[EF_MAX_ARGS] );
      break;

    case 4:
      f4arg  = (void (*)(int *, float *, float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f4arg)( id_ptr, memory + mr_arg_offset_ptr[0], 
		memory + mr_arg_offset_ptr[1], memory + mr_arg_offset_ptr[2], memory + mr_arg_offset_ptr[3], 
		memory + mr_arg_offset_ptr[EF_MAX_ARGS] );
      break;

    case 5:
      f5arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f5arg)( id_ptr, memory + mr_arg_offset_ptr[0], 
		memory + mr_arg_offset_ptr[1], memory + mr_arg_offset_ptr[2], memory + mr_arg_offset_ptr[3], 
		memory + mr_arg_offset_ptr[4], memory + mr_arg_offset_ptr[EF_MAX_ARGS] );
       break;

    case 6:
      f6arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *, 
			 float *))dlsym(ef_ptr->handle, tempText);
      (*f6arg)( id_ptr, memory + mr_arg_offset_ptr[0], 
		memory + mr_arg_offset_ptr[1], memory + mr_arg_offset_ptr[2], memory + mr_arg_offset_ptr[3], 
		memory + mr_arg_offset_ptr[4], memory + mr_arg_offset_ptr[5], 
		memory + mr_arg_offset_ptr[EF_MAX_ARGS+1] );
      break;

    case 7:
      f7arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *, 
			 float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f7arg)( id_ptr, memory + mr_arg_offset_ptr[0], 
		memory + mr_arg_offset_ptr[1], memory + mr_arg_offset_ptr[2], memory + mr_arg_offset_ptr[3], 
		memory + mr_arg_offset_ptr[4], memory + mr_arg_offset_ptr[5], memory + mr_arg_offset_ptr[6],
		memory + mr_arg_offset_ptr[EF_MAX_ARGS] );
      break;

    case 8:
      f8arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *, 
			 float *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f8arg)( id_ptr, memory + mr_arg_offset_ptr[0], 
		memory + mr_arg_offset_ptr[1], memory + mr_arg_offset_ptr[2], memory + mr_arg_offset_ptr[3], 
		memory + mr_arg_offset_ptr[4], memory + mr_arg_offset_ptr[5], memory + mr_arg_offset_ptr[6],
		memory + mr_arg_offset_ptr[7], memory + mr_arg_offset_ptr[EF_MAX_ARGS] );
      break;

    case 9:
      f9arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *, 
			 float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f9arg)( id_ptr, memory + mr_arg_offset_ptr[0], 
		memory + mr_arg_offset_ptr[1], memory + mr_arg_offset_ptr[2], memory + mr_arg_offset_ptr[3], 
		memory + mr_arg_offset_ptr[4], memory + mr_arg_offset_ptr[5], memory + mr_arg_offset_ptr[6],
		memory + mr_arg_offset_ptr[7], memory + mr_arg_offset_ptr[8], 
		memory + mr_arg_offset_ptr[EF_MAX_ARGS] );
      break;

    default:
      fprintf(stderr, "\nNOTICE: External functions with more than %d arguments are not implemented yet.\n\n", EF_MAX_ARGS);
      break;

    }

  } else if ( ef_ptr->internals_ptr->language == EF_C ) {

    fprintf(stderr, "\nERROR: EF_C is not a supported language for External Functions.\n\n");

  }
  
  return;
}





/*
 * Find an external function based on its name and
 * fill in the integer ID associated with that funciton.
 */
int efcn_get_id_( char *name )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;

  static int return_val=0; /* static because it needs to exist after the return statement */

  /*
   * Find the external function.
   */
  status = list_traverse(GLOBAL_ExternalFunctionList, name, EF_ListTraverse_FoundName, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, set the id_ptr to ATOM_NOT_FOUND.
   */
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

  return_val = ef_ptr->id;

  return return_val;
}


/*
 * Find an external function based on a template and
 * fill in the integer ID associated with first function
 * that matches the template.
 */
int efcn_match_template_( char *name )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;

  static int return_val=0; /* static because it needs to exist after the return statement */

  status = list_traverse(GLOBAL_ExternalFunctionList, name, EF_ListTraverse_MatchTemplate, 
			 (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, set the id_ptr to 0
   */
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

  return_val = ef_ptr->id;

  return return_val;
}


/*
 */
void efcn_get_custom_axis_sub_( int *id_ptr, int *axis_ptr, float *lo_ptr, float *hi_ptr, 
			       float *del_ptr, char *unit, int *modulo_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  /*
   * Find the external function.
   */
  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(unit, ef_ptr->internals_ptr->axis[*axis_ptr-1].unit);
  *lo_ptr = ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_lo;
  *hi_ptr = ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_hi;
  *del_ptr = ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_del;
  *modulo_ptr = ef_ptr->internals_ptr->axis[*axis_ptr-1].modulo;

  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the name.
 */
void efcn_get_name_( int *id_ptr, char *name )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(name, ef_ptr->name);

  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the version number.
 */
void efcn_get_version_( int *id_ptr, float *version )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  *version = ef_ptr->internals_ptr->version;

  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the description.
 */
void efcn_get_descr_( int *id_ptr, char *descr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(descr, ef_ptr->internals_ptr->description);

  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the number of arguments.
 */
int efcn_get_num_reqd_args_( int *id_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  static int return_val=0; /* static because it needs to exist after the return statement */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  return_val = ef_ptr->internals_ptr->num_reqd_args;

  return return_val;
}


/*
 * Find an external function based on its integer ID and
 * fill in the flag stating whether the function has
 * a variable number of arguments.
 */
void efcn_get_has_vari_args_( int *id_ptr, int *has_vari_args_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  *has_vari_args_ptr = ef_ptr->internals_ptr->has_vari_args;

  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the axis sources (merged, normal, abstract, custom).
 */
void efcn_get_axis_will_be_( int *id_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_will_be[X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_will_be[Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_will_be[Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_will_be[T_AXIS];

  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the axis sources (merged, normal, abstract, custom).
 */
void efcn_get_piecemeal_ok_( int *id_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->piecemeal_ok[X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->piecemeal_ok[Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->piecemeal_ok[Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->piecemeal_ok[T_AXIS];
  
  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the (boolean) 'axis_implied_from' information for
 * a particular argument to find out if its axes should
 * be merged in to the result grid.
 */
void efcn_get_axis_implied_from_( int *id_ptr, int *iarg_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_implied_from[index][X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_implied_from[index][Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_implied_from[index][Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_implied_from[index][T_AXIS];
  
  
  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the 'arg_extend_lo' information for a particular
 * argument which tells Ferret how much to extend axis limits
 * when providing input data (e.g. to compute a derivative).
 */
void efcn_get_axis_extend_lo_( int *id_ptr, int *iarg_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_extend_lo[index][X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_extend_lo[index][Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_extend_lo[index][Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_extend_lo[index][T_AXIS];
  
  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the 'arg_extend_lo' information for a particular
 * argument which tells Ferret how much to extend axis limits
 * when providing input data (e.g. to compute a derivative).
 */
void efcn_get_axis_extend_hi_( int *id_ptr, int *iarg_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  
  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][T_AXIS];
  
  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the 'arg_extend_lo' information for a particular
 * argument which tells Ferret how much to extend axis limits
 * when providing input data (e.g. to compute a derivative).
 */
void efcn_get_axis_limits_( int *id_ptr, int *axis_ptr, int *lo_ptr, int *hi_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *axis_ptr - 1; /* C indices are 1 less than Fortran */ 

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  
  *lo_ptr = ef_ptr->internals_ptr->axis[index].ss_lo;
  *hi_ptr = ef_ptr->internals_ptr->axis[index].ss_hi;
  
  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the name of a particular argument.
 */
void efcn_get_arg_name_( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 
  int i=0, printable=FALSE;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  
  /*
   * JC_NOTE: if the argument has no name then memory gets overwritten, corrupting
   * the address of iarg_ptr annd causing a core dump.  I need to catch that case
   * here.
   */

  for (i=0;i<strlen(ef_ptr->internals_ptr->arg_name[index]);i++) {
    if (isgraph(ef_ptr->internals_ptr->arg_name[index][i])) {
      printable = TRUE;
      break;
    }
  }

  if ( printable ) {
    strcpy(string, ef_ptr->internals_ptr->arg_name[index]);
  } else {
    strcpy(string, "X");
  }

  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the units for a particular argument.
 */
void efcn_get_arg_unit_( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  
  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
  strcpy(string, ef_ptr->internals_ptr->arg_unit[index]);

  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the description of a particular argument.
 */
void efcn_get_arg_desc_( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  
  strcpy(string, ef_ptr->internals_ptr->arg_desc[index]);

  return;
}



/* .... Object Oriented Utility Functions .... */


/*
 * Allocate space for and initialize the internal
 * information for an EF.
 *
 * Return values:
 *     -1: error allocating space
 *      0: success
 */
int EF_New( ExternalFunction *this )
{
  ExternalFunctionInternals *i_ptr=NULL;
  int status=LIST_OK, i=0, j=0;

  static int return_val=0; /* static because it needs to exist after the return statement */


  /*
   * Allocate space for the internals.
   * If the allocation failed, print a warning message and return.
   */

  this->internals_ptr = malloc(sizeof(ExternalFunctionInternals));
  i_ptr = this->internals_ptr;

  if ( i_ptr == NULL ) {
    fprintf(stderr, "ERROR in EF_New(): cannot allocate ExternalFunctionInternals.\n");
    return_val = -1;
    return return_val;
  }


  /*
   * Initialize the internals.
   */

  /* Information about the overall function */

  i_ptr->version = EF_VERSION;
  strcpy(i_ptr->description, "");
  i_ptr->language = EF_F;
  i_ptr->num_reqd_args = 1;
  i_ptr->has_vari_args = NO;
  for (i=0; i<4; i++) {
    i_ptr->axis_will_be[i] = IMPLIED_BY_ARGS;
    i_ptr->piecemeal_ok[i] = NO;
  }

  /* Information specific to each argument of the function */

  for (i=0; i<EF_MAX_ARGS; i++) {
    for (j=0; j<4; j++) {
      i_ptr->axis_implied_from[i][j] = YES;
      i_ptr->axis_extend_lo[i][j] = 0;
      i_ptr->axis_extend_hi[i][j] = 0;
    }
    strcpy(i_ptr->arg_name[i], "");
    strcpy(i_ptr->arg_unit[i], "");
    strcpy(i_ptr->arg_desc[i], "");
  }

  return return_val;

}
 

/* .... UtilityFunctions for dealing with GLOBAL_ExternalFunctionList .... */

/*
 * Store the global values which will be needed by utility routines
 * in EF_ExternalUtil.c
 */
void EF_store_globals(float *memory_ptr, int *mr_list_ptr, int *cx_list_ptr, 
	int *mres_ptr, float *bad_flag_ptr)
{
  int i=0;

  GLOBAL_memory_ptr = memory_ptr;
  GLOBAL_mr_list_ptr = mr_list_ptr;
  GLOBAL_cx_list_ptr = cx_list_ptr;
  GLOBAL_mres_ptr = mres_ptr;
  GLOBAL_bad_flag_ptr = bad_flag_ptr;

}


/*
 * Generate calls to all of the EF_ExternalUtil.c code in order to 
 * force linking of these routines on Solaris.
 */
void EF_force_linking(int I_should_do_it)
{
  if ( I_should_do_it ) {
    int i = 5;
    float f = 5.0;
    char c = NULL;
    ef_set_num_args_( &i, &i );
    ef_set_has_vari_args_( &i, &i );
    ef_set_piecemeal_ok_( &i, &i, &i, &i, &i );
    ef_set_axis_inheritance_( &i, &i, &i, &i, &i );
    ef_set_axis_influence_( &i, &i, &i, &i, &i, &i );
    ef_set_axis_extend_( &i, &i, &i, &i, &i );

    ef_get_arg_subscripts_( &i, &i, &i, &i );
    ef_get_arg_ss_extremes_( &i, &i, &i );
    ef_get_one_val_( &i, &i, &f );
    ef_get_bad_flags_( &i, &f, &f );

    efcn_get_custom_axis_( &i, &i, &f, &f, &f, &c, &i );
    ef_set_custom_axis_( &i, &i, &f, &f, &f, &c, &i );

    ef_set_desc_( &i, &c);
    ef_set_arg_desc_( &i, &i, &c);
    ef_set_arg_name_( &i, &i, &c);
    ef_set_arg_unit_( &i, &i, &c);

    ef_get_coordinates_( &i, &i, &i, &i, &i, &f );
    ef_get_box_size_( &i, &i, &i, &i, &i, &f );
  }
}


/*
 * Find an external function based on an integer id and return
 * the ef_ptr.
 */
ExternalFunction *ef_ptr_from_id_ptr(int *id_ptr)
{
  static ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nERROR: in ef_ptr_from_id_ptr: No external function of id %d was found.\n\n", *id_ptr);
    return NULL;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
  return ef_ptr;
}


int EF_ListTraverse_fprintf( char *data, char *curr )
{
  FILE *File_ptr=(FILE *)data;
  ExternalFunction *ef_ptr=(ExternalFunction *)curr; 
     
  fprintf(stderr, "path = \"%s\", name = \"%s\", id = %d, internals_ptr = %d\n",
	  ef_ptr->path, ef_ptr->name, ef_ptr->id, ef_ptr->internals_ptr);

  return TRUE;
}
 

/*
 * Ferret always capitalizes everything so we'd better
 * be case INsensitive.
 */
int EF_ListTraverse_FoundName( char *data, char *curr )
{
  ExternalFunction *ef_ptr=(ExternalFunction *)curr; 

  if ( !strcasecmp(data, ef_ptr->name) ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}


int EF_ListTraverse_MatchTemplate( char *data, char *curr )
{
  ExternalFunction *ef_ptr=(ExternalFunction *)curr; 

  int i=0, star_skip=FALSE;
  char *t, *n;

  n = ef_ptr->name;

  for (i=0, t=data; i<strlen(data); i++, t++) {

    if ( *t == '*' ) {

      star_skip = TRUE;
      continue;

    } else if ( *t == '?' ) {

      if ( star_skip ) {
	continue;
      } else {
	if ( ++n == '\0' ) /* end of name */
	  return TRUE; /* no match */
	else
	  continue;
      }

    } else if ( star_skip ) {

      if ( (n = strchr(n, *t)) == NULL ) { /* character not found in rest of name */
	return TRUE; /* no match */
      } else {
	star_skip = FALSE;
      }

    } else if ( *n == '\0' ) /* end of name */
      return TRUE; /* no match */

    else if ( *t == *n )
      continue;

    else
      return TRUE; /* no match */

  } 

  return FALSE; /* got all the way through: a match */

}


int EF_ListTraverse_FoundID( char *data, char *curr )
{
  ExternalFunction *ef_ptr=(ExternalFunction *)curr; 
  int ID=*((int *)data);

  if ( ID == ef_ptr->id ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}


