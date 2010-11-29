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

/* replace_bad_data_sub.c :
/*  replace the missing value if it is either NaN, or if it is equal to */
/*  the old bad value */

/* *kob* - 2/18/99 */
/* v553 *kob* - if the new_bad is NaN, we need to replace any possible
                NaN's in the data, and also swap the old and new bad.  This 
		could happen if use did a set var/bad=nan - new feature */

void replace_bad_data_sub_ ( float *old_bad, float *src, 
			   int *size, float *new_bad )

{
  int i;
  double tmp;
  
  if (isnan(*old_bad)) {
    for (i=0; i<*size; i++)
      if (isnan(src[i])) src[i] = *new_bad;
  } 
  else if (isnan(*new_bad)) {
    for (i=0; i<*size; i++) {
      if (isnan(src[i])) {
	src[i] = *old_bad;
      }
    }
    *new_bad = *old_bad;
  }
  else {
    for (i=0; i<*size; i++) {
      if (src[i] == *old_bad) src[i] = *new_bad;
    }
  }
}


