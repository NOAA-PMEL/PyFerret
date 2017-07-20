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

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
/* *acm   9/06 v600 - add stdlib.h wherever there is stdio.h for altix build*/ 
/* V63  *acm* 10/09 Changes for gfortran build */

#include <Python.h> /* make sure Python.h is first */
#include <stdio.h>
#include <string.h>
#include "fmtprotos.h"

void FORTRAN(tm_make_relative_ver)(char *curr_ver, char *fname, char *path, int *real_ver)
/*
 calling arguments :    
            curr_ver --> contains the relative version num. (eg. .~-3~)
            fname -----> filename; needed for routine high_ver_name
            path ------> path to file, also needed for routine high_ver_name
            real_ver --> will contain and pass back proper version num. (eg. 12 for ~12~)
*/
{
  int i, j, cvlen, high_ver;
  char temp_ver[32];

  /* get just the numeric part of the string, ignoring all else */
  cvlen = strlen(curr_ver);
  for (i=0, j=0; (i < cvlen) && (j < 31); i++) {
      if ( (curr_ver[i] != '.') && (curr_ver[i] != '-') && (curr_ver[i] != '~') ) {
          temp_ver[j] = curr_ver[i];
          j++;
      }
  }
  temp_ver[j] = '\0';

  /* convert the string to an integer */ 
  sscanf(temp_ver, "%d", real_ver);

  /* get the new version number by subtracting the relative version number -1
     from the highest version number          */
  *real_ver -= 1;
  high_ver = high_ver_name(fname, path);
  *real_ver = high_ver - *real_ver;
}

