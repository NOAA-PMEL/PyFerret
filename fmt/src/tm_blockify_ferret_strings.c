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
    The "ferdat" variable is a pointer to an array of string pointers
      where the string pointers are spaced 8 bytes apart
    The buffer pbuff is a long character array. Comy the strings to pbuff
      spacing the strings outstrlen apart.
*/

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include <stdlib.h>

void tm_blockify_ferret_strings(char **mr_blk1, char *pblock,
				int bufsiz, int outstrlen)
{
  int i;
  char *poutchar, *poutstr, *pinchar, **pinstr;

  /* prefill the output buffer with nulls 
     *kob*  fix i<= bufsize bug - corrupted heap */
  for (i=0; i<bufsiz; i++) pblock[i] = 0;

  /* copy all the strings */
  pinstr = mr_blk1;  /* points to each input  string in turn */
  poutstr = pblock;   /* points to each output string in turn */
  for (i=0; i<bufsiz/outstrlen; i++) {
    poutchar = poutstr;             /* point to output characters  */
    pinchar  = *pinstr;              /* point to input  characters  */
    poutstr += outstrlen;           /* point to next output string */
    pinstr += (8/sizeof(char**));   /* point to next input  string */

    /* copy 1 string ... possibly truncated */
    while (poutchar<poutstr && *pinchar) {
      *poutchar = *pinchar;
      poutchar++;
      pinchar ++;
    }

  }

   return;
}
