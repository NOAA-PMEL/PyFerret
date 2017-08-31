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



/*
*	ferret_dispatch_c - C interface routine to set up structure arguments
*	                    for calling ferret_dispatch.F
*
* TMAP interactive data analysis program
*
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*/

/*
* revision history:
*   11/16/94 - updated to use macro declarations from ferret_shared_buffer.h
*
*   05/25/95 - added ifdef check for trailing underscores not needed on HP *kob*
*   *js* 6.99 Set line buffering if in server mode
*  *acm* 1/12 - Ferret 6.8 ifdef double_p for double-precision ferret, see the
*              definition of macro DFTYPE in ferret.h 
*  *sh* 1/17 - trac enhancement #2369 -- dynamic memory management 
*/

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "ferret.h"
#include "ferret_shared_buffer.h"

void ferret_dispatch_c(char *init_command, smPtr sBuffer)
{
  int flag_buff_size  = NUMFLAGS;
  int TEXTLENGTH_size  = TEXTLENGTH;

/* call the FORTRAN program that actually does the FERRET command */
/* all arguments must be pointers for FORTRAN */
/*ifdef check added 5/95 *kob* */
/* 1/17 *sh* removed the "memory" argument -- replaced by dynamic allocation
   of hyperslab memory using FORTRAN90 pointers and c (or python) */

  FORTRAN(ferret_dispatch)(init_command, sBuffer->flags, &flag_buff_size,
		    sBuffer->text, &TEXTLENGTH_size, &(sBuffer->numStrings));

  return;
}

static int SecureFlag = 0;
static int ServerFlag = 0;

/*
 * Routines for setting/getting security settings
 */
void set_secure(void) {
  SecureFlag = 1;
}

int FORTRAN(is_secure)(void) {
  return SecureFlag;
}
/*
 * Routines for setting/getting server settings
 */
void set_server(void) {
  ServerFlag = 1;
  /* Should always be line buffered */
  setvbuf(stdout, NULL, _IOLBF, 0);
  setvbuf(stderr, NULL, _IOLBF, 0);
}

int FORTRAN(is_server)(void) {
  return ServerFlag;
}

