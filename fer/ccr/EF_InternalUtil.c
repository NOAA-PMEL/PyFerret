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

static LIST *GLOBAL_ExternalFunctionList;
static int I_have_scanned_already = FALSE;
static int I_have_warned_already = FALSE;


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

int efcn_gather_info_( int * );
void efcn_get_axis_abstract_( int *, float *, int *, int *, int *, int *, int * );
void efcn_get_axis_custom_( int * );

void efcn_compute( int *, int *, int *, int *, float *, int *, int *, float *);

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
void efcn_get_arg_name_( int *, int *, char * );
void efcn_get_arg_units_( int *, int *, char * );
void efcn_get_arg_descr_( int *, int *, char * );


/* ... Functions called internally .... */

void EF_gather_fcn_internals( ExternalFunction * );

int  EF_ListTraverse_fprintf( char *, char * );
int  EF_ListTraverse_FoundName( char *, char * );
int  EF_ListTraverse_MatchTemplate( char *, char * );
int  EF_ListTraverse_FoundID( char *, char * );


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

  /* static because it needs to exist after the return statement */
  static int return_val=0;

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
   * - Get all the paths from the "FER_EF" environment variable.
   *
   * - While there is another path:
   *    - get the path;
   *    - create a pipe for the "ls -1" command;
   *    - read stdout and use each file name to create another external function entry;
   *
   */

  if ( !getenv("FER_EF") ) {
    if ( !I_have_warned_already ) {
      fprintf(stderr, "\nWARNING: environment variable FER_EF not defined.\n\n");
      I_have_warned_already = TRUE;
    }
    return_val = 0;
    return return_val;
  }

  sprintf(paths, "%s", getenv("FER_EF"));
    
  path = strtok(paths, " \t");
    
  if (path[strlen(path)-1] != '/')
    strcat(path, "/"); 

  if ( path == NULL ) {
 
    fprintf(stderr, "\nWARNING:No paths were found in the environment variable FER_EF\n\n");

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

  /*
   * debugging line
   *
   status = list_traverse(GLOBAL_ExternalFunctionList, stderr, EF_ListTraverse_fprintf, (LIST_FRNT | LIST_FORW | LIST_ALTR));
   *
   */


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

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    return;
  }

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
  int status=LIST_OK, i=0;
  char ef_object[1024]="", tempText[EF_MAX_NAME_LENGTH]="", *c;

  void *handle;
  void (*langfptr)(int *);
  int (*fptr)(int, void *);
  void (*finfoptr)(float *, char (*)[EF_MAX_DESCRIPTION_LENGTH], int *, int *, int (*)[4], int (*)[4]);
  void (*farginfoptr)(int *, int (*)[4], int (*)[4], int (*)[4], char (*)[EF_MAX_NAME_LENGTH],
		      char (*)[EF_MAX_NAME_LENGTH], char (*)[EF_MAX_DESCRIPTION_LENGTH]);

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING in efcn_gather_info(): No external function of id %d was found.\n\n", *id_ptr);
    return -1;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

  if ( (ef_ptr->internals_ptr = malloc(sizeof(ExternalFunctionInternals))) == NULL ) {
    fprintf(stderr, "ERROR in efcn_gather_info(): cannot allocate ExternalFunctionInternals.\n");
    return -1;
  }

  strcat(ef_object, ef_ptr->path);
  strcat(ef_object, ef_ptr->name);
  strcat(ef_object, ".so");

  strcat(tempText, ef_ptr->name);
  strcat(tempText, "_lang_");

  ef_ptr->handle = dlopen(ef_object, RTLD_LAZY);
  
  langfptr  = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
  if (langfptr == NULL) {
    fprintf(stderr, "ERROR in efcn_gather_info(): %s\n", dlerror());
    return -1;
  }

  (*langfptr)(&(ef_ptr->language));

  if ( ef_ptr->language == EF_C ) {

    fptr  = (int (*)(int, void *))dlsym(ef_ptr->handle, ef_ptr->name);
    if (fptr == NULL) {
      fprintf(stderr, "ERROR in efcn_gather_info(): %s\n", dlerror());
      return -1;
    }

    (*fptr)(CONFIGURE, (void *)ef_ptr);

  } else if ( ef_ptr->language == EF_F ) {

    ExternalFunctionInternals *ptr = ef_ptr->internals_ptr;
    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_get_info_");
    finfoptr  = (void (*)(float *, char (*)[128], int *, int *, int (*)[4], int (*)[4]))dlsym(ef_ptr->handle, tempText);
    if (finfoptr == NULL) {
      fprintf(stderr, "ERROR in efcn_gather_info(): %s\n", dlerror());
      return -1;
    }

    (*finfoptr)(&(ptr->version), &(ptr->description), &(ptr->num_reqd_args),
		&(ptr->has_vari_args), &(ptr->axis_will_be), &(ptr->piecemeal_ok) );

    /* trim white space at the end */
    c = &(ptr->description[EF_MAX_DESCRIPTION_LENGTH-1]); 
    while ( isspace(*c) ) { c--; } *(++c) = '\0';

    for (i=0; i<ptr->num_reqd_args; i++) {
      int j = i+1;
      sprintf(tempText, "");
      strcat(tempText, ef_ptr->name);
      strcat(tempText, "_get_arg_info_");
      farginfoptr = (void (*)(int *, int (*)[4], int (*)[4], int (*)[4], char (*)[EF_MAX_NAME_LENGTH],
		     char (*)[EF_MAX_NAME_LENGTH], char (*)[EF_MAX_DESCRIPTION_LENGTH]))dlsym(ef_ptr->handle, tempText);
      if (farginfoptr == NULL) {
	fprintf(stderr, "ERROR in efcn_gather_info(): %s\n", dlerror());
	return -1;
      }

      (*farginfoptr)( &j, &(ptr->axis_implied_from[i]), &(ptr->axis_extend_lo[i]), &(ptr->axis_extend_hi[i]), 
		      &(ptr->arg_name[i]), &(ptr->arg_units[i]), &(ptr->arg_descr[i]) );

      /* trim white space at the end */
      c = &(ptr->arg_name[i][EF_MAX_NAME_LENGTH-1]); 
      while ( isspace(*c) ) { c--; } *(++c) = '\0';

      c = &(ptr->arg_units[i][EF_MAX_NAME_LENGTH-1]); 
      while ( isspace(*c) ) { c--; } *(++c) = '\0';

      c = &(ptr->arg_descr[i][EF_MAX_DESCRIPTION_LENGTH-1]); 
      while ( isspace(*c) ) { c--; } *(++c) = '\0';
    }

  }
  
  return 0;
}


/*
 * Find an external function based on its integer ID, 
 * Query the function about an axis defined as abstract and ask
 * for the low and high subscripts on that axis. Pass memory,
 * mr_list and cx_list info into the external function.
 */
void efcn_get_axis_abstract_( int *id_ptr, float *memory_ptr, int *mr_list_ptr, int *cx_list_ptr,
			      int *iaxis_ptr, int *loss_ptr, int *hiss_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;
  char tempText[EF_MAX_NAME_LENGTH]="";

  int dummy = 19;

  void (*fptr)(float *, int *, int *, int *, int *, int *);

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  
  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

  if ( ef_ptr->language == EF_F ) {

    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_get_axis_abstract_");

    fptr  = (void (*)(float *, int *, int *, int *, int *, int *))dlsym(ef_ptr->handle, tempText);
    (*fptr)( memory_ptr, mr_list_ptr, cx_list_ptr, iaxis_ptr, loss_ptr, hiss_ptr);

  } else {

    fprintf(stderr, "\nExternal Functions in C are not supported yet.\n\n");

  }

  return;
}


/*
 */
void efcn_get_axis_custom_( int *id_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

  /* JC_TODO: write code for efcn_get_axis_custom. */
  fprintf(stderr, "\nNo code written for this routine yet.\n\n");

  return;
}


/*
 * Find an external function based on its integer ID, 
 * pass the necessary information and the data and tell
 * the function to calculate the result.
 */
void efcn_compute_( int *id_ptr, int *narg_ptr, int *loss_ptr, 
		    int *hiss_ptr, float *bad_flag_ptr,
		    int *mr_arg_offset_ptr, int *mr_res_offset_ptr, float *memory )
{
  ExternalFunction *ef_ptr=NULL;
  ComputeInfo compute_info;
  int status=LIST_OK, xyzt=0, i=0;
  int arg_points[EF_MAX_ARGS];
  char tempText[EF_MAX_NAME_LENGTH]="";

  int (*fptr)(int, void *);
  void (*f1arg)(float *, float *, float *);
  void (*f2arg)(float *, float *, float *, float *);
  void (*f3arg)(float *, float *, float *, float *, float *);
  void (*f4arg)(float *, float *, float *, float *, float *, float *);

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 


  if ( ef_ptr->language == EF_F ) {

    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_compute_");

    switch ( ef_ptr->internals_ptr->num_reqd_args ) {

    case 1:
      f1arg  = (void (*)(float *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f1arg)( bad_flag_ptr, memory + mr_arg_offset_ptr[0], memory + mr_arg_offset_ptr[1]);
      break;

    case 2:
      f2arg  = (void (*)(float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f2arg)( bad_flag_ptr, memory + mr_arg_offset_ptr[0], memory + mr_arg_offset_ptr[1], 
		memory + mr_arg_offset_ptr[2]);
      break;

    case 3:
      f3arg  = (void (*)(float *, float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f3arg)( bad_flag_ptr, memory + mr_arg_offset_ptr[0], memory + mr_arg_offset_ptr[1],
		memory + mr_arg_offset_ptr[2], memory + mr_arg_offset_ptr[3]);
      break;

    case 4:
      f4arg  = (void (*)(float *, float *, float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
      (*f4arg)( bad_flag_ptr, memory + mr_arg_offset_ptr[0], memory + mr_arg_offset_ptr[1],
		memory + mr_arg_offset_ptr[2], memory + mr_arg_offset_ptr[3], memory + mr_arg_offset_ptr[4]);
      break;

    default:
      fprintf(stderr, "\nNOTICE: External functions with more than 4 arguments are not implemented yet.\n\n");
      break;

    }

  } else if ( ef_ptr->language == EF_C ) {

    /*
     * Copy the data passed by reference to the 'ComputeInfo' structure.
     */

    compute_info.narg = *narg_ptr;
    compute_info.num_datapoints = 0;

    for (i=0; i<*narg_ptr; i++) {
      arg_points[i] = 1;
      for (xyzt=0; xyzt<4; xyzt++) {
	compute_info.loss[i][xyzt] = loss_ptr[xyzt];
	compute_info.hiss[i][xyzt] = hiss_ptr[xyzt];
	arg_points[i] *= (1 + compute_info.hiss[i][xyzt] - compute_info.loss[i][xyzt]);
      }
      compute_info.num_datapoints += arg_points[i];
      compute_info.bad_flag[i] = bad_flag_ptr[i];
      compute_info.data[i] = memory + mr_arg_offset_ptr[i];
    }

    compute_info.bad_flag[*narg_ptr] = bad_flag_ptr[*narg_ptr]; /* this is the bad flag for the result */
    compute_info.data[*narg_ptr] = memory + mr_arg_offset_ptr[*narg_ptr]; /* this is the memory address for the result */

    fptr  = (int (*)(int, void *))dlsym(ef_ptr->handle, ef_ptr->name);

    (*fptr)(COMPUTE, (void *)(&compute_info));

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
 * Find an external function based on its integer ID and
 * fill in the name.
 */
void efcn_get_name_( int *id_ptr, char *name )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    strcpy(name, "");
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

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
  int status=LIST_OK;

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    *version = 0.0;
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

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
  int status=LIST_OK;

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    strcpy(descr, "");
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

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
  int status=LIST_OK;

  static int return_val=0; /* static because it needs to exist after the return statement */

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    return_val = 0;
    return return_val;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList);

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
  int status=LIST_OK;

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    *has_vari_args_ptr = 0;
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

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
  int status=LIST_OK;

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    *array_ptr = 0;
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

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
  int status=LIST_OK;

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    *array_ptr = 0;
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
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
  int status=LIST_OK;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    *array_ptr = 0;
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
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
  int status=LIST_OK;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    *array_ptr = 0;
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
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
  int status=LIST_OK;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    *array_ptr = 0;
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][T_AXIS];
  
  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the name of a particular argument.
 */
void efcn_get_arg_name_( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    strcpy(string, "");
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
  strcpy(string, ef_ptr->internals_ptr->arg_name[index]);

  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the units for a particular argument.
 */
void efcn_get_arg_units_( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    strcpy(string, "");
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
  strcpy(string, ef_ptr->internals_ptr->arg_units[index]);

  return;
}


/*
 * Find an external function based on its integer ID and
 * fill in the description of a particular argument.
 */
void efcn_get_arg_descr_( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nWARNING: No external function of id %d was found.\n\n", *id_ptr);
    strcpy(string, "");
    return;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
  strcpy(string, ef_ptr->internals_ptr->arg_descr[index]);

  return;
}



/* .... UtilityFunctions for dealing with GLOBAL_ExternalFunctionList .... */


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


