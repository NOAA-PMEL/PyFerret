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

#include <stdio.h>
#include <stdlib.h>

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

