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



/*	ferret_query - C routine to query state information from FERRET 

* TMAP interactive data analysis program

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*/

/*
* revision history:
   11/16/94 - updated to use macro declarations from ferret_shared_buffer.h
   
   05/25/95 - added ifdef check for trailing underscores not needed on HP *kob*

*/

/*
compile this with
   cc -c -I/home/rascal/oz/Ferret_gui ferret_query.c
   (and use -D_NO_PROTO for non-ANSI compilers)
*/ 

#include <stdio.h>
#include <stdlib.h>

#include "ferret_shared_buffer.h"

/* function prototype for FORTRAN routine */
/* added ifdef for necessity of trailing underscores *kob* */

#ifdef NO_ENTRY_NAME_UNDERSCORES
#    ifdef _NO_PROTO
void ferret_query_f( );
#    else
void ferret_query_f( int *query, int *flags, int *nflags,
		      char *text, int *ntext,
		      int *numStrings, int *numNumbers,
		      double *nums, int *ncoord,
		      char *arg1, char *arg2, char *arg3,
		      char *arg4, char *arg5,
		      int *status );
#    endif
#else                              /*NO_ENTRY_NAME_UNDERSCORES*/
#    ifdef _NO_PROTO
void ferret_query_f_( );
#    else
void ferret_query_f_( int *query, int *flags, int *nflags,
		      char *text, int *ntext,
		      int *numStrings, int *numNumbers,
		      double *nums, int *ncoord,
		      char *arg1, char *arg2, char *arg3,
		      char *arg4, char *arg5,
		      int *status );
#    endif
#endif                             /*NO_ENTRY_NAME_UNDERSCORES*/

#ifdef _NO_PROTO
int ferret_query(query, sBuffer, arg1, arg2, arg3, arg4, arg5 )
int query;
smPtr sBuffer;
char *arg1, *arg2, *arg3, *arg4, *arg5;

#else
int ferret_query(int query, smPtr sBuffer,
		 char *arg1, char *arg2, char *arg3, char *arg4, char *arg5 )
#endif
{
  int flag_buff_size  = NUMFLAGS;
  int TEXTLENGTH_size  = TEXTLENGTH;
  int NUMDOUBLES_size = NUMDOUBLES;

  int status, i;

/* diagnostic code */
#ifdef QUERY_DEBUG
  printf("Query number %d\n",query);
  if ( arg1[0] ) printf("Arg 1 = %s\n",arg1);
  if ( arg2[0] ) printf("Arg 2 = %s\n",arg2);
  if ( arg3[0] ) printf("Arg 3 = %s\n",arg3);
  if ( arg4[0] ) printf("Arg 4 = %s\n",arg4);
  if ( arg5[0] ) printf("Arg 5 = %s\n",arg4);
  *(sBuffer->text) = 'Q';
#endif

/* call the FORTRAN program that actually does the query */
/* all arguments must be pointers for FORTRAN */

#ifdef NO_ENTRY_NAME_UNDERSCORES
  ferret_query_f( &query, sBuffer->flags, &flag_buff_size,
#else
  ferret_query_f_( &query, sBuffer->flags, &flag_buff_size,
#endif
		   sBuffer->text, &TEXTLENGTH_size,
		   &(sBuffer->numStrings), &(sBuffer->numNumbers),
		   &(sBuffer->nums[0]), &NUMDOUBLES_size,
		   arg1, arg2, arg3, arg4, arg5, &status );


/* diagnostic code */
#ifdef QUERY_DEBUG
  for (i=0; i<NUMFLAGS; i++) printf("Flag %d is %d\n",i,sBuffer->flags[i]);
  {
    char *ptext = sBuffer->text;
    while (*ptext) {
      putchar((int) *ptext);
      if (13 == (int) *ptext ) putchar('\n');
      ptext++;
    }
  }
/*  printf("Returned text:%s\n",sBuffer->text); */
  printf("numStrings: %d\n",sBuffer->numStrings);
  printf("numNumbers: %d\n",sBuffer->numNumbers);
  for (i=0; i<sBuffer->numNumbers; i++) printf("Object %d is %g\n",i,
					      sBuffer->nums[i]);
#endif

  return( status );
}

