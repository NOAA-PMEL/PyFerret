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
   Return (copy) the null-terminated string to the array provided
*/

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include <stdlib.h>
#include <string.h>

void get_offset_c_string_(fer_ptr, offset, outstring, maxlen)
     char*** fer_ptr;
     int* offset;
     char* outstring;
     int* maxlen;
{
  char** each_str_ptr;
  char* str_ptr;
  int i=0;

  each_str_ptr = *fer_ptr;   /* holds pointer to the first string */
  each_str_ptr += *offset * 8/sizeof(char**); /* point to the desired strng */ 
  str_ptr = *each_str_ptr;

  /* copy the characters of the string, itself, up to maxlen */
  while (i<*maxlen && *str_ptr!=0)
    {
      outstring[i++] = *(str_ptr++);
    }

  /* pad the remainder with blanks */
  while (i<*maxlen)
    {
      outstring[i++] = ' ';
    }

  return;
}




