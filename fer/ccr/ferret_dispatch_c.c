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



/*	ferret_dispatch_c - C interface routine to set up structure arguments
	                    for calling ferret_dispatch.F

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
   cc -c -I/home/rascal/oz/Ferret_gui ferret_dispatch_c.c
   (and use -D_NO_PROTO for non-ANSI compilers)
*/ 

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <ferret.h>
#include "ferret_shared_buffer.h"

/* function prototype for FORTRAN routine */
/* added ifdef for necessity of trailing underscores *kob* */

#ifdef NO_ENTRY_NAME_UNDERSCORES
#   ifdef _NO_PROTO
void ferret_dispatch_( );
#   else
void ferret_dispatch( float *memory, char *init_command, int *rtn_flags,
		       int *nflags, char *rtn_chars, int *nchars, int *nerrlines );
#   endif
#else                     /*NO_ENTRY_NAME_UNDERSCORES*/
#   ifdef _NO_PROTO
void ferret_dispatch_( );
#   else
void ferret_dispatch_( float *memory, char *init_command, int *rtn_flags,
		       int *nflags, char *rtn_chars, int *nchars, int *nerrlines );
#   endif
#endif                    /*NO_ENTRY_NAME_UNDERSCORES*/

#ifdef _NO_PROTO
void ferret_dispatch_c( memory, init_command, sBuffer )
float *memory;
char *init_command;
smPtr sBuffer;
#else
void ferret_dispatch_c( float *memory, char *init_command, smPtr sBuffer )
#endif
{
  int flag_buff_size  = NUMFLAGS;
  int TEXTLENGTH_size  = TEXTLENGTH;
  int NUMDOUBLES_size = NUMDOUBLES;

/* call the FORTRAN program that actually does the FERRET command */
/* all arguments must be pointers for FORTRAN */
/*ifdef check added 5/95 *kob* */

#ifdef NO_ENTRY_NAME_UNDERSCORES
  ferret_dispatch( memory, init_command, sBuffer->flags, &flag_buff_size,
#else
  ferret_dispatch_( memory, init_command, sBuffer->flags, &flag_buff_size,
#endif
		   sBuffer->text, &TEXTLENGTH_size, &(sBuffer->numStrings) );

  return;
}

static int SecureFlag = 0;
static int ServerFlag = 0;

/*
 * Routines for setting/getting security settings
 */
void set_secure() {
  SecureFlag = 1;
}

int FORTRAN(is_secure)() {
  return SecureFlag;
}
/*
 * Routines for setting/getting server settings
 */
void set_server() {
  ServerFlag = 1;
}

int FORTRAN(is_server)() {
  return ServerFlag;
}


