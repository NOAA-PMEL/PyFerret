/*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.
*
*/



/* EF_InternalUtil.c
 *
 * Jonathan Callahan
 * Sep 4th 1997
 *
 * This file contains all the utility functions which Ferret
 * needs in order to communicate with an external function.
 */


/* .................... Includes .................... */
 
#include <unistd.h>		/* for convenience */
#include <stdio.h>		/* for convenience */
#include <stdlib.h>		/* for convenience */
#include <string.h>		/* for convenience */
#include <fcntl.h>		/* for fcntl() */
#include <dlfcn.h>		/* for dynamic linking */
#include <signal.h>             /* for signal() */
#include <setjmp.h>             /* required for jmp_buf */

#include <sys/types.h>	        /* required for some of our prototypes */
#include <sys/stat.h>
#include <sys/errno.h>

#include "EF_Util.h"
#include "list.h"  /* locally added list library */


/* ................ Global Variables ................ */
/*
 * The memory_ptr, mr_list_ptr and cx_list_ptr are obtained from Ferret
 * and cached whenever they are passed into one of the "efcn_" functions.
 * These pointers can be accessed by the utility functions in efn_ext/.
 * This way the EF writer does not need to see these pointers.
 */

static LIST  *GLOBAL_ExternalFunctionList;
float *GLOBAL_memory_ptr;
int   *GLOBAL_mr_list_ptr;
int   *GLOBAL_cx_list_ptr;
int   *GLOBAL_mres_ptr;
float *GLOBAL_bad_flag_ptr;

/*
 * The jumpbuffer is used by setjmp() and longjmp().
 * setjmp() is called by FORTRAN(efcn_compute)() in EF_InternalUtil.c and
 * saves the stack environment in jumpbuffer for later use by longjmp().
 * This allows one to bail out of external functions and still
 * return control to Ferret.
 * Check "Advanced Progrmming in the UNIX Environment" by Stevens
 * sections 7.10 and 10.14 to understand what's going on with these.
 */
static jmp_buf jumpbuffer;
static sigjmp_buf sigjumpbuffer;
static volatile sig_atomic_t canjump;

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

int  FORTRAN(efcn_scan)( int * );
int  FORTRAN(efcn_already_have_internals)( int * );

int  FORTRAN(efcn_gather_info)( int * );
void FORTRAN(efcn_get_custom_axes)( int *, int * );
void FORTRAN(efcn_get_result_limits)( int *, float *, int *, int * );
void FORTRAN(efcn_compute)( int *, int *, int *, int *, int *, float *, int *, float *, int * );


void FORTRAN(efcn_get_custom_axis_sub)( int *, int *, float *, float *, float *, char *, int * );

int  FORTRAN(efcn_get_id)( char * );
int  FORTRAN(efcn_match_template)( int *, char * );

void FORTRAN(efcn_get_name)( int *, char * );
void FORTRAN(efcn_get_version)( int *, float * );
void FORTRAN(efcn_get_descr)( int *, char * );
int  FORTRAN(efcn_get_num_reqd_args)( int * );
void FORTRAN(efcn_get_has_vari_args)( int *, int * );
void FORTRAN(efcn_get_axis_will_be)( int *, int * );
void FORTRAN(efcn_get_axis_reduction)( int *, int * );
void FORTRAN(efcn_get_piecemeal_ok)( int *, int * );

void FORTRAN(efcn_get_axis_implied_from)( int *, int *, int * );
void FORTRAN(efcn_get_axis_extend_lo)( int *, int *, int * );
void FORTRAN(efcn_get_axis_extend_hi)( int *, int *, int * );
void FORTRAN(efcn_get_axis_limits)( int *, int *, int *, int * );
int  FORTRAN(efcn_get_arg_type)( int *, int *);
void FORTRAN(efcn_get_arg_name)( int *, int *, char * );
void FORTRAN(efcn_get_arg_unit)( int *, int *, char * );
void FORTRAN(efcn_get_arg_desc)( int *, int *, char * );


/* .... Functions called internally .... */

/* Fortran routines from the efn/ directory */
void FORTRAN(efcn_copy_array_dims)(void);
void FORTRAN(efcn_get_workspace_addr)(float *, int *, float *);

static void EF_signal_handler(int);
static void (*fpe_handler)(int);
static void (*segv_handler)(int);
static void (*int_handler)(int);
static void (*bus_handler)(int);


void FORTRAN(ef_err_bail_out)(int *, char *);

void EF_store_globals(float *, int *, int *, int *, float *);

ExternalFunction *ef_ptr_from_id_ptr(int *);

int  EF_ListTraverse_fprintf( char *, char * );
int  EF_ListTraverse_FoundName( char *, char * );
int  EF_ListTraverse_MatchTemplate( char *, char * );
int  EF_ListTraverse_FoundID( char *, char * );

int  EF_New( ExternalFunction * );


/* .............. Function Definitions .............. */


/* .... Functions for use by Ferret (to be called from Fortran) .... */

/*
 * Note that all routines called directly from Ferret,
 * ie. directly from Fortran, should be all lower case,
 * should pass by reference and should end with an underscore.
 */

/*
 * Find all of the ~.so files in directories listed in the
 * FER_EXTERNAL_FUNCTIONS environment variable and add all 
 * the names and associated directory information to the 
 * GLOBAL_ExternalFunctionList.
 */
int FORTRAN(efcn_scan)( int *gfcn_num_internal )
{
  
  FILE *file_ptr=NULL;
  ExternalFunction ef; 
 
  char file[EF_MAX_NAME_LENGTH]="";
  char *path_ptr=NULL, path[8192]="";
  char paths[8192]="", cmd[EF_MAX_DESCRIPTION_LENGTH]="";
  int count=0, status=LIST_OK;

  static int return_val=0; /* static because it needs to exist after the return statement */

  if ( I_have_scanned_already ) {
    return_val = list_size(GLOBAL_ExternalFunctionList);
    return return_val;
  }

  if ( (GLOBAL_ExternalFunctionList = list_init()) == NULL ) {
    fprintf(stderr, "ERROR: efcn_scan: Unable to initialize GLOBAL_ExternalFunctionList.\n");
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
      fprintf(stderr, "\
\nWARNING: environment variable FER_EXTERNAL_FUNCTIONS not defined.\n\n");
      I_have_warned_already = TRUE;
    }
    return_val = 0;
    return return_val;
  }

  sprintf(paths, "%s", getenv("FER_EXTERNAL_FUNCTIONS"));
    
  path_ptr = strtok(paths, " \t");

  if ( path_ptr == NULL ) {
 
    fprintf(stderr, "\
\nWARNING:No paths were found in the environment variable FER_EXTERNAL_FUNCTIONS.\n\n");

    return_val = 0;
    return return_val;
 
  } else {
    
    do {

	  strcpy(path, path_ptr);

      if (path[strlen(path)-1] != '/')
        strcat(path, "/"); 

      sprintf(cmd, "ls -1 %s", path);

      /* Open a pipe to the "ls" command */
      if ( (file_ptr = popen(cmd, "r")) == (FILE *) NULL ) {
	    fprintf(stderr, "\nERROR: Cannot open pipe.\n\n");
	    return_val = -1;
	    return return_val;
      }
 
      /*
       * Read a line at a time.
       * Any ~.so files are assumed to be external functions.
       */
      while ( fgets(file, EF_MAX_NAME_LENGTH, file_ptr) != NULL ) {

        char *extension;

	    file[strlen(file)-1] = '\0';   /* chop off the carriage return */
	    extension = &file[strlen(file)-3];
	    if ( strcmp(extension, ".so") == 0 ) {
          file[strlen(file)-3] = '\0'; /* chop off the ".so" */
	      strcpy(ef.path, path);
	      strcpy(ef.name, file);
	      ef.id = *gfcn_num_internal + ++count; /* pre-increment because F arrays start at 1 */
	      ef.already_have_internals = NO;
	      ef.internals_ptr = NULL;
	      list_insert_after(GLOBAL_ExternalFunctionList, &ef, sizeof(ExternalFunction));
	    }

      }
 
      pclose(file_ptr);
 
      path_ptr = strtok(NULL, " \t"); /* get the next directory */
 
    } while ( path_ptr != NULL );

    I_have_scanned_already = TRUE;
  }

  return_val = count;
  return return_val;

}


/*
 * Determine whether an external function has already 
 * had its internals read.
 */
int FORTRAN(efcn_already_have_internals)( int *id_ptr )
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
int FORTRAN(efcn_gather_info)( int *id_ptr )
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

  /*
   * Following might need to be added to get WebFerret to work
   * Commented out because it's Solaris specific
   * JS
   */
#if 0
  if (dlopen("liballferret.so", RTLD_LAZY | RTLD_GLOBAL) == NULL){
      fprintf(stderr, "  dlerror: %s\n", dlerror());
  }
  if (dlopen(ef_object, RTLD_LAZY | RTLD_GLOBAL) == NULL){
      fprintf(stderr, "  dlerror: %s\n", dlerror());
  }

  ef_ptr->handle = dlopen(0, RTLD_LAZY);
#endif
  /*  if ( (ef_ptr->handle = dlopen(ef_object, RTLD_LAZY)) == NULL ) {*/
  /* if ( (ef_ptr->handle = dlopen(ef_object, RTLD_NOW || RTLD_GLOBAL)) == NULL ) { */
  /* kob - commented out above line, and removed RTLD_GBAL check from below on
     advice of jc - osf didn't have a definition for RTLD_GLOBAL */
  if ( (ef_ptr->handle = dlopen(ef_object, RTLD_NOW)) == NULL ) {
    fprintf(stderr, "\n\
ERROR in External Function %s:\n\
Dynamic linking call dlopen() returns --\n\
\"%s\".\n", ef_ptr->name, dlerror());
    return -1;
  }
  
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

    /*
     * Prepare for bailout possibilities by setting a signal handler for
     * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack 
     * environment with sigsetjmp (for the signal handler) and setjmp 
     * (for the "bail out" utility function).
     */   
    if ( (fpe_handler = signal(SIGFPE, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGFPE.\n");
      return;
    }
    if ( (segv_handler = signal(SIGSEGV, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGSEGV.\n");
      return;
    }
    if ( (int_handler = signal(SIGINT, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGINT.\n");
      return;
    }
    if ( (bus_handler = signal(SIGBUS, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGBUS.\n");
      return;
    }
    if (sigsetjmp(sigjumpbuffer, 1) != 0) {
      /* Warning message printed by signal handler. */
      return;
    }
    canjump = 1;

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

    /*
     * Restore the old signal handlers.
     */
    if (signal(SIGFPE, (*fpe_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGFPE handler.\n");
      return;
    }
    if (signal(SIGSEGV, (*segv_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGSEGV handler.\n");
      return;
    }
    if (signal(SIGINT, (*int_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGINT handler.\n");
      return;
    }
    if (signal(SIGBUS, (*bus_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGBUS handler.\n");
      return;
    }

  }
  
  return 0;
}


/*
 * Find an external function based on its integer ID, 
 * Query the function about custom axes. Store the context
 * list information for use by utility functions.
 */
void FORTRAN(efcn_get_custom_axes)( int *id_ptr, int *cx_list_ptr )
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

    /*
     * Prepare for bailout possibilities by setting a signal handler for
     * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack 
     * environment with sigsetjmp (for the signal handler) and setjmp 
     * (for the "bail out" utility function).
     */   
    if ( (fpe_handler = signal(SIGFPE, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGFPE.\n");
      return;
    }
    if ( (segv_handler = signal(SIGSEGV, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGSEGV.\n");
      return;
    }
    if ( (int_handler = signal(SIGINT, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGINT.\n");
      return;
    }
    if ( (bus_handler = signal(SIGBUS, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGBUS.\n");
      return;
    }
    if (sigsetjmp(sigjumpbuffer, 1) != 0) {
      /* Warning message printed by signal handler. */
      return;
    }
    canjump = 1;

    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_custom_axes_");

    fptr  = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
    (*fptr)( id_ptr );

    /*
     * Restore the old signal handlers.
     */
    if (signal(SIGFPE, (*fpe_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGFPE handler.\n");
      return;
    }
    if (signal(SIGSEGV, (*segv_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGSEGV handler.\n");
      return;
    }
    if (signal(SIGINT, (*int_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGINT handler.\n");
      return;
    }
    if (signal(SIGBUS, (*bus_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGBUS handler.\n");
      return;
    }

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
void FORTRAN(efcn_get_result_limits)( int *id_ptr, float *memory, int *mr_list_ptr, int *cx_list_ptr )
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

    /*
     * Prepare for bailout possibilities by setting a signal handler for
     * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack 
     * environment with sigsetjmp (for the signal handler) and setjmp 
     * (for the "bail out" utility function).
     */   
    if ( (fpe_handler = signal(SIGFPE, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGFPE.\n");
      return;
    }
    if ( (segv_handler = signal(SIGSEGV, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGSEGV.\n");
      return;
    }
    if ( (int_handler = signal(SIGINT, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGINT.\n");
      return;
    }
    if ( (bus_handler = signal(SIGBUS, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGBUS.\n");
      return;
    }
    if (sigsetjmp(sigjumpbuffer, 1) != 0) {
      /* Warning message printed by signal handler. */
      return;
    }
    canjump = 1;

    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_result_limits_");

    fptr  = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
    (*fptr)( id_ptr);

    /*
     * Restore the old signal handlers.
     */
    if (signal(SIGFPE, (*fpe_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGFPE handler.\n");
      return;
    }
    if (signal(SIGSEGV, (*segv_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGSEGV handler.\n");
      return;
    }
    if (signal(SIGINT, (*int_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGINT handler.\n");
      return;
    }
    if (signal(SIGBUS, (*bus_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGBUS handler.\n");
      return;
    }

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
void FORTRAN(efcn_compute)( int *id_ptr, int *narg_ptr, int *cx_list_ptr, int *mr_list_ptr, int *mres_ptr,
	float *bad_flag_ptr, int *mr_arg_offset_ptr, float *memory, int *status )
{
  ExternalFunction *ef_ptr=NULL;
  ExternalFunctionInternals *i_ptr=NULL;
  float *arg_ptr[EF_MAX_COMPUTE_ARGS];
  int xyzt=0, i=0, j=0;
  int size=0;
  char tempText[EF_MAX_NAME_LENGTH]="";

  /*
   * Prototype all the functions needed for varying numbers of
   * arguments and work arrays.
   */

  void (*fptr)(int *);
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
  void (*f10arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *);
  void (*f11arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *);
  void (*f12arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *);
  void (*f13arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *);
  void (*f14arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *,
        float *);
  void (*f15arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *);
  void (*f16arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *);
  void (*f17arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *);
  void (*f18arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *);

  /*
   * Initialize the status
   */
  *status = FERR_OK;

  /*
   * Store the array dimensions for memory resident variables and for working storage.
   * Store the memory pointer and various lists globally.
   */
  FORTRAN(efcn_copy_array_dims)();
  EF_store_globals(memory, mr_list_ptr, cx_list_ptr, mres_ptr, bad_flag_ptr);

  /*
   * Find the external function.
   */
  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) {
    fprintf(stderr, "\n\
ERROR in efcn_compute() finding external function: id = [%d]\n", *id_ptr);
    *status = FERR_EF_ERROR;
    return;
  }

  i_ptr = ef_ptr->internals_ptr;

  if ( i_ptr->language == EF_F ) {

    /*
     * Begin assigning the arg_ptrs.
     */

    /* First come the arguments to the function. */

     for (i=0; i<i_ptr->num_reqd_args; i++) {
       arg_ptr[i] = memory + mr_arg_offset_ptr[i];
     }

    /* Now for the result */

     arg_ptr[i++] = memory + mr_arg_offset_ptr[EF_MAX_ARGS];

    /* Now for the work arrays */

    /*
     * If this program has requested working storage we need to 
     * ask the function to specify the amount of space needed
     * and then create the memory here.  Memory will be released
     * after the external function returns.
     */
    if (i_ptr->num_work_arrays > EF_MAX_WORK_ARRAYS) {

	  fprintf(stderr, "\n\
ERROR specifying number of work arrays in ~_init subroutine of external function %s\n\
\tnum_work_arrays[=%d] exceeds maximum[=%d].\n\n", ef_ptr->name, i_ptr->num_work_arrays, EF_MAX_WORK_ARRAYS);
	  *status = FERR_EF_ERROR;
	  return;

    } else if (i_ptr->num_work_arrays < 0) {

	  fprintf(stderr, "\n\
ERROR specifying number of work arrays in ~_init subroutine of external function %s\n\
\tnum_work_arrays[=%d] must be a positive number.\n\n", ef_ptr->name, i_ptr->num_work_arrays);
	  *status = FERR_EF_ERROR;
	  return;

    } else if (i_ptr->num_work_arrays > 0)  {

      sprintf(tempText, "");
      strcat(tempText, ef_ptr->name);
      strcat(tempText, "_work_size_");

      fptr = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
      if (fptr == NULL) {
	fprintf(stderr, "\n\
ERROR in efcn_compute() accessing %s\n", tempText);
	*status = FERR_EF_ERROR;
        return;
      }
      (*fptr)( id_ptr );

      for (j=0; j<i_ptr->num_work_arrays; i++, j++) {

        size = sizeof(float);
        for (xyzt=0; xyzt<4; xyzt++) {
          size *= i_ptr->work_array_len[j][xyzt];
        }

	/* Allocate memory for each individual work array */
        if ( (arg_ptr[i] = (float *)malloc(size)) == NULL ) { 
          fprintf(stderr, "\n\
ERROR in efcn_compute() allocating %d words of memory\n", size);
	  *status = FERR_EF_ERROR;
	  return;
        }
      }

    }

    /*
     * Prepare for bailout possibilities by setting a signal handler for
     * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack 
     * environment with sigsetjmp (for the signal handler) and setjmp 
     * (for the "bail out" utility function).
     */   
    if ( (fpe_handler = signal(SIGFPE, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGFPE.\n");
      *status = FERR_EF_ERROR;
      return;
    }
    if ( (segv_handler = signal(SIGSEGV, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGSEGV.\n");
      *status = FERR_EF_ERROR;
      return;
    }
    if ( (int_handler = signal(SIGINT, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGINT.\n");
      *status = FERR_EF_ERROR;
      return;
    }
    if ( (bus_handler = signal(SIGBUS, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in efcn_gather_info() catching SIGBUS.\n");
      *status = FERR_EF_ERROR;
      return;
    }
    if (sigsetjmp(sigjumpbuffer, 1) != 0) {
      /* Warning message printed by signal handler. */
      *status = FERR_EF_ERROR;
      return;
    }
    canjump = 1;

    if (setjmp(jumpbuffer) != 0 ) {
      /* Warning message printed by bail-out utility function. */
      *status = FERR_EF_ERROR;
      return;
    }


    /*
     * Now go ahead and call the external function's "_compute_" function,
     * prototyping it for the number of arguments expected.
     */
    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_compute_");

    switch ( i_ptr->num_reqd_args + i_ptr->num_work_arrays ) {

    case 1:
	  f1arg  = (void (*)(int *, float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f1arg)( id_ptr, arg_ptr[0], arg_ptr[1] );
	break;


    case 2:
	  f2arg  = (void (*)(int *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f2arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2] );
	break;


    case 3:
	  f3arg  = (void (*)(int *, float *, float *, float *, float *))
        dlsym(ef_ptr->handle, tempText);
	  (*f3arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3] );
	break;


    case 4:
	  f4arg  = (void (*)(int *, float *, float *, float *, float *, float *))
        dlsym(ef_ptr->handle, tempText);
	  (*f4arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4] );
	break;


    case 5:
	  f5arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *))
        dlsym(ef_ptr->handle, tempText);
	  (*f5arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5] );
	break;


    case 6:
	  f6arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *))dlsym(ef_ptr->handle, tempText);
	  (*f6arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6] );
	break;


    case 7:
	  f7arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f7arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7] );
	break;


    case 8:
	  f8arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f8arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8] );
	break;


    case 9:
	  f9arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f9arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9] );
	break;


    case 10:
	  f10arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f10arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10] );
	break;


    case 11:
	  f11arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f11arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11] );
	break;


    case 12:
	  f12arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *, float *, float *))
        dlsym(ef_ptr->handle, tempText);
	  (*f12arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12] );
	break;


    case 13:
	  f13arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *, float *, float *, float *))
        dlsym(ef_ptr->handle, tempText);
	  (*f13arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13] );
	break;


    case 14:
	  f14arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *, float *, float *, float *,
        float *))dlsym(ef_ptr->handle, tempText);
	  (*f14arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14] );
	break;


    case 15:
	  f15arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f15arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15] );
	break;


    case 16:
	  f16arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f16arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15], arg_ptr[16] );
	break;


    case 17:
	  f17arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f17arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15], arg_ptr[16],
        arg_ptr[17] );
	break;


    case 18:
	  f18arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
	  (*f18arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15], arg_ptr[16],
        arg_ptr[17], arg_ptr[18] );
	break;


    default:
      fprintf(stderr, "\n\
ERROR: External functions with more than %d arguments are not implemented yet.\n\n", EF_MAX_ARGS);
      *status = FERR_EF_ERROR;
      return;
      break;

    }

    /*
     * Restore the old signal handlers.
     */
    if (signal(SIGFPE, (*fpe_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGFPE handler.\n");
      *status = FERR_EF_ERROR;
      return;
    }
    if (signal(SIGSEGV, (*segv_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGSEGV handler.\n");
      *status = FERR_EF_ERROR;
      return;
    }
    if (signal(SIGINT, (*int_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGINT handler.\n");
      *status = FERR_EF_ERROR;
      return;
    }
    if (signal(SIGBUS, (*bus_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in efcn_gather_info() restoring default SIGBUS handler.\n");
      *status = FERR_EF_ERROR;
      return;
    }

    /*
     * Now it's time to release the work space.
     * With arg_ptr[0] for argument #1, and remembering one slot for the result,
     * we should begin freeing up memory at arg_ptr[num_reqd_args+1].
     */
    for (i=i_ptr->num_reqd_args+1; i<i_ptr->num_reqd_args+1+i_ptr->num_work_arrays; i++) {
      free(arg_ptr[i]);
    }

  } else if ( ef_ptr->internals_ptr->language == EF_C ) {

    fprintf(stderr, "\n\
ERROR: External Functions may not yet be written in C.\n\n");
    *status = FERR_EF_ERROR;
    return;

  }
  

  return;
}


/*
 * A signal handler for SIGFPE, SIGSEGV, SIGINT and SIGBUS signals generated
 * while executing an external function.  See "Advanced Programming
 * in the UNIX Environment" p. 299 ff for details.
 */
static void EF_signal_handler(int signo) {

  if (canjump == 0) return; /* unexpected signal, ignore */

  /*
   * Restore the old signal handlers.
   */
  if (signal(SIGFPE, (*fpe_handler)) == SIG_ERR) {
    fprintf(stderr, "\nERROR in EF_signal_handler() restoring old SIGFPE handler.\n");
    return;
  }
  if (signal(SIGSEGV, (*segv_handler)) == SIG_ERR) {
    fprintf(stderr, "\nERROR in EF_signal_handler() restoring old SIGSEGV handler.\n");
	return;
  }
  if (signal(SIGINT, (*int_handler)) == SIG_ERR) {
    fprintf(stderr, "\nERROR in EF_signal_handler() restoring old SIGINT handler.\n");
    return;
  }
  if (signal(SIGBUS, (*bus_handler)) == SIG_ERR) {
    fprintf(stderr, "\nERROR in EF_signal_handler() restoring old SIGBUS handler.\n");
    return;
  }

  if (signo == SIGFPE) {
    fprintf(stderr, "\n\nERROR in external function: Floating Point Error\n");
    canjump = 0;
    siglongjmp(sigjumpbuffer, 1);
  } else if (signo == SIGSEGV) {
    fprintf(stderr, "\n\nERROR in external function: Segmentation Violation\n");
    canjump = 0;
    siglongjmp(sigjumpbuffer, 1);
  } else if (signo == SIGINT) {
    fprintf(stderr, "\n\nExternal function halted with Control-C\n");
    canjump = 0;
    siglongjmp(sigjumpbuffer, 1);
  } else if (signo == SIGBUS) {
    fprintf(stderr, "\n\nERROR in external function: Hardware Fault\n");
    canjump = 0;
    siglongjmp(sigjumpbuffer, 1);
  } else {
    fprintf(stderr, "\n\nERROR in external function: signo = %d\n", signo);
    canjump = 0;
    siglongjmp(sigjumpbuffer, 1);
  }

}


/*
 * Find an external function based on its name and
 * return the integer ID associated with that funciton.
 */
int FORTRAN(efcn_get_id)( char name[] )
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
 * Determine whether a function name matches a template.
 * Return 1 if the name matchs.
 */
int FORTRAN(efcn_match_template)( int *id_ptr, char template[] )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;
  int EF_LT_MT_return;

  static int return_val=0; /* static because it needs to exist after the return statement */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return return_val; }

  EF_LT_MT_return = EF_ListTraverse_MatchTemplate((char *)template, (char *)ef_ptr);
  
  /* The list package forces 'list traversal' functions to return
   * 0 whenever a match is found.  We want to return a more reasonable
   * 1 (=true) if we find a match.
   */
  if ( EF_LT_MT_return == FALSE ) {
	return_val = 1;
  } else {
    return_val = 0;
  }

  return return_val;
}


/*
 */
void FORTRAN(efcn_get_custom_axis_sub)( int *id_ptr, int *axis_ptr, float *lo_ptr, float *hi_ptr, 
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
 * return the name.
 */
void FORTRAN(efcn_get_name)( int *id_ptr, char *name )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(name, ef_ptr->name);

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the version number.
 */
void FORTRAN(efcn_get_version)( int *id_ptr, float *version )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  *version = ef_ptr->internals_ptr->version;

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the description.
 */
void FORTRAN(efcn_get_descr)( int *id_ptr, char *descr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(descr, ef_ptr->internals_ptr->description);

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the number of arguments.
 */
int FORTRAN(efcn_get_num_reqd_args)( int *id_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  static int return_val=0; /* static because it needs to exist after the return statement */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  return_val = ef_ptr->internals_ptr->num_reqd_args;

  return return_val;
}


/*
 * Find an external function based on its integer ID and
 * return the flag stating whether the function has
 * a variable number of arguments.
 */
void FORTRAN(efcn_get_has_vari_args)( int *id_ptr, int *has_vari_args_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  *has_vari_args_ptr = ef_ptr->internals_ptr->has_vari_args;

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the axis sources (merged, normal, abstract, custom).
 */
void FORTRAN(efcn_get_axis_will_be)( int *id_ptr, int *array_ptr )
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
 * return the axis_reduction (retained, reduced) information.
 */
void FORTRAN(efcn_get_axis_reduction)( int *id_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_reduction[X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_reduction[Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_reduction[Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_reduction[T_AXIS];

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the piecemeal_ok information.  This lets Ferret
 * know if it's ok to break up a calculation along an axis
 * for memory management reasons.
 */
void FORTRAN(efcn_get_piecemeal_ok)( int *id_ptr, int *array_ptr )
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
 * return the (boolean) 'axis_implied_from' information for
 * a particular argument to find out if its axes should
 * be merged in to the result grid.
 */
void FORTRAN(efcn_get_axis_implied_from)( int *id_ptr, int *iarg_ptr, int *array_ptr )
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
 * return the 'arg_extend_lo' information for a particular
 * argument which tells Ferret how much to extend axis limits
 * when providing input data (e.g. to compute a derivative).
 */
void FORTRAN(efcn_get_axis_extend_lo)( int *id_ptr, int *iarg_ptr, int *array_ptr )
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
 * return the 'arg_extend_hi' information for a particular
 * argument which tells Ferret how much to extend axis limits
 * when providing input data (e.g. to compute a derivative).
 */
void FORTRAN(efcn_get_axis_extend_hi)( int *id_ptr, int *iarg_ptr, int *array_ptr )
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
 * return the 'axis_limits' information for a particular
 * argument.
 */
void FORTRAN(efcn_get_axis_limits)( int *id_ptr, int *axis_ptr, int *lo_ptr, int *hi_ptr )
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
 * return the 'arg_type' information for a particular
 * argument which tells Ferret whether an argument is a 
 * float or a string.
 */
int FORTRAN(efcn_get_arg_type)( int *id_ptr, int *iarg_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  static int return_val=0; /* static because it needs to exist after the return statement */
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  
  return_val = ef_ptr->internals_ptr->arg_type[index];
  
  return return_val;
}


/*
 * Find an external function based on its integer ID and
 * return the name of a particular argument.
 */
void FORTRAN(efcn_get_arg_name)( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 
  int i=0, printable=FALSE;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  
  /*
   * JC_NOTE: if the argument has no name then memory gets overwritten, corrupting
   * the address of iarg_ptr and causing a core dump.  I need to catch that case
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
    strcpy(string, "--");
  }

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the units for a particular argument.
 */
void FORTRAN(efcn_get_arg_unit)( int *id_ptr, int *iarg_ptr, char *string )
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
 * return the description of a particular argument.
 */
void FORTRAN(efcn_get_arg_desc)( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  
  strcpy(string, ef_ptr->internals_ptr->arg_desc[index]);

  return;
}



void FORTRAN(ef_err_bail_out)(int *id_ptr, char *text)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  fprintf(stderr, "\n\
Bailing out of external function \"%s\":\n\
\t%s\n", ef_ptr->name, text);

  longjmp(jumpbuffer, 1);
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
  i_ptr->num_work_arrays = 0;
  for (i=0; i<4; i++) {
    for (j=0; j<EF_MAX_WORK_ARRAYS; j++) {
      i_ptr->work_array_len[j][i] = 1;
    }
    i_ptr->axis_will_be[i] = IMPLIED_BY_ARGS;
    i_ptr->axis_reduction[i] = RETAINED;
    i_ptr->piecemeal_ok[i] = NO;
  }

  /* Information specific to each argument of the function */

  for (i=0; i<EF_MAX_ARGS; i++) {
    for (j=0; j<4; j++) {
      i_ptr->axis_implied_from[i][j] = YES;
      i_ptr->axis_extend_lo[i][j] = 0;
      i_ptr->axis_extend_hi[i][j] = 0;
    }
    i_ptr->arg_type[i] = FLOAT_ARG;
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


int EF_ListTraverse_MatchTemplate( char data[], char *curr )
{
  ExternalFunction *ef_ptr=(ExternalFunction *)curr; 

  int i=0, star_skip=FALSE;
  char upname[EF_MAX_DESCRIPTION_LENGTH];
  char *t, *n;

  for (i=0; i<strlen(ef_ptr->name); i++) {
    upname[i] = toupper(ef_ptr->name[i]);
  }
  upname[i] = '\0';

  n = upname;

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

    else if ( *t == *n ) {
      n++;
      continue;
    }

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


