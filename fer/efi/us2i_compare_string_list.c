

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

#ifdef MAC_SSIZE
typedef long ssize_t;
#endif

#include <stdio.h>
#include <string.h>
#include "list.h"  /* locally added list library */

/* max length of a path */
#define MAX_NAME 512

/* define structure used locally */
typedef struct  {
    char astring[MAX_NAME];
    int seq;
} strngs;

#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif

/* .... Functions called by Fortran .... */
void FORTRAN(us2i_compare_string_list)(char *, int *);
void FORTRAN(us2i_str_cmp)(char *, char *, int *);

/* .... Static Variables ............... */
static LIST *GLOBAL_unique_us2i_List;
static int us2i_list_initialized = 0;

/* .... Functions called internally .... */
static int add_us2i_string(char *);
static int ListTraverse_FoundString( char *, char * );


/* ----
 * Deallocate GLOBAL_unique_us2i_List
 *
 *
 * int FORTRAN(end_us2i_list)
 * {
 *    list_free(GLOBAL_unique_us2i_List, LIST_DEALLOC);
 *    us2i_list_initialized = 0;
 * }
 */


/* ----
 * Call C strcmp function.
 */
void FORTRAN(us2i_str_cmp)(char *str1, char *str2, int *ival)
{
   *ival = strcmp(str1, str2);
}


/* ----
 * Find a string in the list, initializing the list if this is the
 * first search. If the string is not in the list, add it to the list.
 * Return the sequence number of the string in the (resulting) list.
 * If an error occurs, a sequence number of zero is assigned.
 */
void FORTRAN(us2i_compare_string_list)(char* compare_string, int *str_seq)
{
   strngs *str_ptr;
   int status;

   if ( ! us2i_list_initialized ) {
      /*
       * no list yet; initialize the list and add the string to it;
       * send back the new sequence number of this string
       */
      *str_seq = add_us2i_string(compare_string);
      return;
   }

   /* check the existing list for this string */
   status = list_traverse(GLOBAL_unique_us2i_List, compare_string, 
                          ListTraverse_FoundString, (LIST_FRNT | LIST_FORW | LIST_ALTR));
   if ( status != LIST_OK ) {
      /* string not found; add it to the list and send back the new sequence number */
      *str_seq = add_us2i_string(compare_string); 
      return;
   }

   /* String found; get and send back its sequence number in the list */
   str_ptr = (strngs *) list_curr(GLOBAL_unique_us2i_List); 
   *str_seq = str_ptr->seq;
}


/* ----
 * Add a string to GLOBAL_unique_us2i_List, initializing the list if necessary.
 * Returns the sequence number of this new string, or 0 if an error occurs.
 */
static int add_us2i_string(char addstring[])
{
   strngs this_string;
   int isize;
   int iseq;

   /* Create the list if required */
   if ( ! us2i_list_initialized ) {
      GLOBAL_unique_us2i_List = list_init();
      if ( GLOBAL_unique_us2i_List == NULL ) {
         fprintf(stderr, "ERROR: unique_str2int: Unable to initialize GLOBAL_unique_us2i_List.\n");
         return 0;
      }
      us2i_list_initialized = 1;
   }

   /* Add to global linked list*/ 
   isize = list_size(GLOBAL_unique_us2i_List);
   iseq = 1 + isize;

   this_string.seq = iseq;
   strcpy(this_string.astring, addstring);

   list_insert_after(GLOBAL_unique_us2i_List, (char *) &this_string, sizeof(this_string));

   return iseq;
}


/* ---- 
 * See if the incoming string matches the string in the list at curr.
 * Case sensitive.
 */
static int ListTraverse_FoundString(char *data, char *curr)
{
   strngs *str_ptr = (strngs *) curr; 

   if ( strcmp(data, str_ptr->astring) == 0 ) {
      return 0; /* found match */
   }
   return 1;
}

