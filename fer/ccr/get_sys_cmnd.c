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
 * execute the passed command string and append the lines of input to the
 * array of strings supplied
 * V530  9/00 *sh*
 * V740  3/18 *acm* issue #1862, results may not end with a newline
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ferret.h"
#include "FerMem.h"

void FORTRAN(get_sys_cmnd)(char ***fer_ptr, int *nlines, char *cmd, int *stat)
/*
     char*** fer_ptr; output: char** pointer to strings 
     int* nlines; output: number of strings read 
     char* cmd; input: the shell command to execute 
     int* stat;
*/
{
    char** sarray;
    int linebufsize =  BUFSIZ;  /* initial size of input line buffer */
    char* buf;
    char* newbuf;
    char** newsarray;
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

    /*
     * Use calloc for sarray to initialize everything to NULL pointers
     * for Ferret's string arrays.
     */
    sarray = (char **) FerMem_Malloc(BUFSIZ * sizeof(char *), __FILE__, __LINE__);
    if ( sarray == NULL ) {
       *stat = 1;
       return;
    }
    memset(sarray, 0, BUFSIZ * sizeof(char *));

    buf = (char *) FerMem_Malloc(sizeof(char) * linebufsize, __FILE__, __LINE__);
    if ( buf == NULL ) {
       FerMem_Free(sarray, __FILE__, __LINE__);
       *stat = 1;
       return;
    }

    fpipe = popen(cmd, "r");
    if ( fpipe != NULL ) {

       /* read one input line
	      previously assumed if there was no newline the line was incomplete
		  Now check to see if we overran the buffer, else return the line as is*/
       while ( fgets(buf, linebufsize, fpipe) != NULL ) {
          slen = strlen(buf);
          incomplete = buf[slen-1] != '\n';
		  if (incomplete && slen <= linebufsize-2)
		  {
			 buf[slen+1] = '\n';
			 incomplete = 0;
			 slen++;
		  }
          if (incomplete) {
             /* line buffer wasn't large enough --> allocate more */ 
			 while (incomplete) {
                linebufsize += BUFSIZ;
                newbuf = (char *) FerMem_Realloc(buf, sizeof(char) * linebufsize, __FILE__, __LINE__);
                if ( newbuf == NULL ) {
                   FerMem_Free(buf, __FILE__, __LINE__);
                   for (i = 0; i < *nlines; i++)
                      FerMem_Free(sarray[i], __FILE__, __LINE__);
                   FerMem_Free(sarray, __FILE__, __LINE__);
                   *stat = 1;
                   return;
                }
                buf = newbuf;
                if (fgets(buf+slen, BUFSIZ, fpipe) != 0) {
                   slen = strlen(buf);
                   incomplete = buf[slen-1] != '\n';
                }
                else
                   incomplete = 0;
             }
          }
          buf[slen-1] = 0;  /* remove newline */

          /* make and save a permanent copy of the input line */
          /* BUG FIX *kob* v552 - need to add one to string length for null */
          pmnt = (char *) FerMem_Malloc(sizeof(char) * (int)(strlen(buf)+1), __FILE__, __LINE__);
          if ( pmnt == NULL ) {
             FerMem_Free(buf, __FILE__, __LINE__);
             for (i = 0; i < *nlines; i++)
                FerMem_Free(sarray[i], __FILE__, __LINE__);
             FerMem_Free(sarray, __FILE__, __LINE__);
             *stat = 1;
             return;
          }
          strcpy(pmnt, buf);
          if ( nincr == last_increment ) {
             /* double the length of the string pointer array */
             last_increment = increment;
             increment *= 2;
             newsarray = (char **) FerMem_Realloc(sarray, sizeof(char *) * increment, __FILE__, __LINE__);
             if ( newsarray == NULL ) {
                FerMem_Free(buf, __FILE__, __LINE__);
                for (i = 0; i < *nlines; i++)
                   FerMem_Free(sarray[i], __FILE__, __LINE__);
                FerMem_Free(sarray, __FILE__, __LINE__);
                FerMem_Free(pmnt, __FILE__, __LINE__);
                *stat = 1;
                return;
             }
             sarray = newsarray;
             /* Initialize new string pointer to NULL for Ferret's string arrays. */
             for (i = *nlines; i < increment; i++)
                sarray[i] = NULL;
             nincr = 0;
          }
          sarray[(*nlines)++] = pmnt; 
          nincr++;
       }

       /* done with the pipe */
       pclose(fpipe);
    }

    /* buf no longer needed */
    FerMem_Free(buf, __FILE__, __LINE__);
    buf = NULL;

    /* always return at least one string (avoid FORTRAN probs) */
    /* *kob* v552 - bug fix - still need to allocate space for the null string */
    if (*nlines == 0 ) {
       pmnt = (char *) FerMem_Malloc(sizeof(char), __FILE__, __LINE__);
       if ( pmnt == NULL ) {
          FerMem_Free(sarray, __FILE__, __LINE__);
          *stat = 1;
          return;
       }
       *pmnt = 0;
       sarray[0] = pmnt;
       *nlines = 1;
    }

    /* Return the char** pointer */
    *fer_ptr = sarray;
    return;
}

