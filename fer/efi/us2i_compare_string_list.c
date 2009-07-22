

/*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granteHd the
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

/* us2i_compare_string_list.c
 *
 * Ansley Manke
 * Uses list-handling library to create a list of unique names for the
 * function unique_str2int
 */


#include <wchar.h>
#include <unistd.h>		/* for convenience */
#include <stdlib.h>		/* for convenience */
#include <stdio.h>		/* for convenience */
#include <string.h>		/* for convenience */
#include <fcntl.h>		/* for fcntl() */
#include <assert.h>
#include <sys/types.h>	        /* required for some of our prototypes */
#include <sys/stat.h>
#include <sys/errno.h>

#include "list.h"  /* locally added list library */
#include "us2i_compare_string_list.h"

/* ................ Global Variables ................ */

static LIST  *GLOBAL_unique_us2i_List;
static int us2i_list_initialized = FALSE;

/* ............. Function Declarations .............. */
/*
 * Note that all routines called directly from Ferret,
 * ie. directly from Fortran, should be all lower case,
 * be of type 'void', pass by reference and should end with 
 * an underscore.
 */


/* .... Functions called by Fortran .... */
int FORTRAN(us2i_compare_string_list) (char *, int *);
int FORTRAN(init_us2i_list)(char *);
int FORTRAN(end_us2i_list);
void FORTRAN(us2i_str_cmp)(char *, char *, int *);

/* .... Functions called internally .... */

int (add_us2i_string)(char *);

int ListTraverse_FoundString( char *, char * );
void list_free(LIST *, int ); 

/* ----
 * Initialize new list of strings, GLOBAL_unique_us2i_List
 */

int FORTRAN(init_us2i_list)(char *str1)

{
  strngs this_string; 
  int iseq;
  static int return_val=3; /* static because it needs to exist after the return statement */

/* Add string to global string linked list*/ 
  if (!us2i_list_initialized) {
    if ( (GLOBAL_unique_us2i_List = list_init()) == NULL ) {
      fprintf(stderr, "ERROR: unique_str2int: Unable to initialize GLOBAL_unique_us2i_List.\n");
      return_val = -1;
      return return_val; 
	}
    us2i_list_initialized = TRUE;
  }

  strcpy(this_string.astring, str1);
  iseq = 1;
  this_string.seq = iseq;

  list_insert_after(GLOBAL_unique_us2i_List, &this_string, sizeof(strngs));
  return_val = 3;
  return return_val;
  }

/* ----
 * Deallocate GLOBAL_unique_us2i_List
 */
/*  */
/* int FORTRAN(end_us2i_list) */
/*  */
/* { */
/*  */
/*   static int return_val=3; /* static because it needs to exist after the return statement */
/*  */
/*   list_free(GLOBAL_unique_us2i_List, LIST_DEALLOC); */
/*   return return_val;  */
/*   } */

/* ----
 * Add a string to GLOBAL_unique_us2i_List
 */

int add_us2i_string(char addstring[])

{
  strngs this_string;
  int iseq;
  strngs *str_ptr=NULL;
  
  int isize;

  static int return_val=3; /* static because it needs to exist after the return statement */


	/* Add to global linked list*/ 
  if (!us2i_list_initialized) {
    if ( (GLOBAL_unique_us2i_List = list_init()) == NULL ) {
      fprintf(stderr, "ERROR: unique_str2int: Unable to initialize GLOBAL_unique_us2i_List.\n");
      return_val = -1;
      return return_val; 
		}
    us2i_list_initialized = TRUE;
  }
  
  isize = list_size(GLOBAL_unique_us2i_List);
  iseq = 1 + isize;

  this_string.seq = iseq;
  strcpy(this_string.astring, addstring);

  list_insert_after(GLOBAL_unique_us2i_List, &this_string, sizeof(this_string));
  
  return return_val;
}


/* ----
 * Call C strcmp function.
 */
void FORTRAN(us2i_str_cmp)(char *str1, char *str2, int *ival)

{
  static int return_val=0;
  *ival = strcmp(str1, str2);

  return;
}

/* ----
 * Find a string in the list. If it is not in the list, add it to the list.
 * If it is in the list, return its sequence number.
 */
 int FORTRAN(us2i_compare_string_list) (char* compare_string, int *str_seq)

{
  strngs *str_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  LIST *dummy;


   /*
   * Check the list of strings for this string.  If not found, add it, and
   * send back the sequence number.
   */  
	   
  status = list_traverse(GLOBAL_unique_us2i_List, compare_string, ListTraverse_FoundString, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    add_us2i_string(compare_string); 
	  status = list_traverse(GLOBAL_unique_us2i_List, compare_string, ListTraverse_FoundString, (LIST_FRNT | LIST_FORW | LIST_ALTR));
	  str_ptr = (strngs *)list_curr(GLOBAL_unique_us2i_List);
	  *str_seq = str_ptr->seq;

    return return_val;
  }


   /*
   * If found, Send back the corresponding sequence number.
   */  

  str_ptr = (strngs *)list_curr(GLOBAL_unique_us2i_List); 
  *str_seq = str_ptr->seq;
  return_val = 3;
  return return_val;
}


/* ---- 
 * See if the incoming string matches the string in the list at 
 * curr. Ferret always capitalizes everything so be case INsensitive.
 */
int ListTraverse_FoundString( char *data, char *curr )
{
  strngs *str_ptr=(strngs *)curr; 

  if ( !strcmp(data, str_ptr->astring) ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}
