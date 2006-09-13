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
*/

/*
  execute the passed command string and append the lines of input to the
  array of strings supplied
  V530  9/00 *sh*
*/

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
/* *acm   9/06 v600 - add stdlib.h wherever there is stdio.h for altix build*/ 
#include <wchar.h>
#include <stdlib.h>
#include <stdio.h>

void get_sys_cmnd_(fer_ptr, nlines, cmd, stat)
     char*** fer_ptr; /* output: char** pointer to strings */
     int* nlines; /* output: number of strings read */
     char* cmd; /* input: the shell command to execute */
     int* stat;
  {
    char** sarray;
    int linebufsize =  BUFSIZ;  /* initial size of input line buffer */
    char* buf;
    FILE *fpipe;
    char* pmnt;
    int nincr  = 0;  /* lines read in in this increment of the sarray */
    int i, slen;
    int incomplete;  /* if buffer is too small for some input line */
    int increment = BUFSIZ;  /* extend length of char** ptr next by this */
    int last_increment = increment;

    /* initialize */
    *nlines = 0;
    *stat = 0;
    if ( !(sarray = (char **) malloc(sizeof(char *) * BUFSIZ)) ) 
      {
	*stat = 1;
	return;
      }
    if ( !(buf = (char *) malloc(sizeof(char) * linebufsize)) ) 
      {
	*stat = 1;
	return;
      }

    if ((fpipe = popen(cmd, "r")) != NULL)

      /* read one newline-terminated input line */
      while (fgets(buf, linebufsize, fpipe) != NULL)
	{
	  slen = strlen(buf);
	  incomplete = buf[slen-1] != '\n';
	  if (incomplete) 
	    {
	    /* line buffer wasn't large enough --> allocate more */
	    while (incomplete) {
	      linebufsize += BUFSIZ;
	      if ( !(buf = (char *) realloc((void*)buf,
					    sizeof(char) * linebufsize)) )
		{
		  *stat = 1;
		  return;
		}
	      if (fgets(buf+slen, BUFSIZ, fpipe) != 0) {
		slen = strlen(buf);
		incomplete = buf[slen-1] != '\n';
	      } else
		incomplete = 0;
	    }
	  }
	  buf[slen-1] = 0;  /* remove newline */

	  /* make and save a permanent copy of the input line */
	  /* BUG FIX *kob* v552 - need to add one to string 
	     length for newline   */
	  if ( !(pmnt = (char *) malloc(sizeof(char) * (int)(strlen(buf)+1))))
	    {
	      *stat = 1;
	      return;
	    }
	  strcpy(pmnt, buf);
	  if (nincr == last_increment)
	  /* double the length of the string pointer array */
	    {
	      last_increment = increment;
	      increment *= 2;
	      if ( !(sarray = (char **) realloc( (void*) sarray,
						 sizeof(char *) * increment ) ))
		{
		  *stat = 1;
		  return;
		}
	      nincr = 0;
	    }
	  sarray[(*nlines)++] = pmnt; 
	  nincr++;
	 }

    /* done with the pipe */
    pclose(fpipe);

    /* always return at least one string (avoid FORTRAN probs) */
    /* *kob* v552 - bug fix - still need to allocate space for the null string */
    if (*nlines == 0 ) 
      {
	if ( !(pmnt = (char *) malloc(sizeof(char) )))
	  {
	    *stat = 1;
	    return;
	  }
	*pmnt = 0;
	
	sarray[0] = pmnt;
	(*nlines)++;
      }

    /* Return the char** pointer */
    *fer_ptr = sarray;

    /* free temporary dynamic memory */
    free(buf);

    return;
  }

