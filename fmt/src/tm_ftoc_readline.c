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



/* tm_ftoc_readline -- based on "manexamp.c" in the readline distribution. */
/* c jacket  to make gnu readline callable from FORTRAN */

/* had to add ifdef check for trailing underscore in routine name
   for aix port *kob* 10/94 */

/* Readline is very slow for piped I/O, so run w/o readline for Ferret server
 * *js* 12/98
 */

/* v51 *kob* - upgraded to new version of readline, which is now seperate
   from tmap library - modified readline include to that end */

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include <stdio.h>
#include <strings.h>
#include "readline.h"

/* Easier way of handling FORTRAN calls with underscore/no underscore */
#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif


/* A static variable for holding the line. */
static char *line_read = (char *)NULL;

/* Read a string, and return a pointer to it.  Returns NULL on EOF. */
char *do_gets ( prompt )
  char *prompt;

{
  /* If the buffer has already been allocated, return the memory
     to the free pool. */
  if (line_read != (char *)NULL)
    {
      free (line_read);
      line_read = (char *)NULL;
    }

  /* Get a line from the user. */
  /* If running in server mode, don't use fancy readline stuff */

  if (!FORTRAN(is_server)()){
    line_read = readline (prompt);
  } else {
    char* loc;
    fputs(prompt, stdout);
    fflush(stdout);
    line_read = (char *)malloc(2048);
    fgets(line_read, 2047, stdin);
    loc = rindex(line_read, '\n');
    if (loc != 0){
      *loc = '\0';
    }
  }

  /* If the line has any text in it, save it on the history. */
  if (line_read && *line_read)
    add_history (line_read);

  return (line_read);
}

#ifdef NO_ENTRY_NAME_UNDERSCORES
tm_ftoc_readline( prompt, buff )
#else
tm_ftoc_readline_( prompt, buff )
#endif
/* c jacket routine to make gnu readline callable from FORTRAN */
  char *prompt, *buff;
{
  char *ptr;

/* invoke gnu readline with line recall and editing */
  ptr = do_gets ( prompt );

/* copy the line into the buffer provided from FORTRAN */
  if (ptr != (char *)NULL)
    strcpy( buff, ptr );
  else
    buff[0] = '\004';   /* ^D  */

  return (0);
}
