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
 * Return a substring of a given length starting from offset
 * of given string.
 * 
 * If offset is greater than full string length, return an empty string.  If offset is less then
 * full string length, but offset+substring length is greater than full string lenght, just
 * return rest of full string in substring.  
 *
 * V541: *kob* 3/02
 * V680  *acm* 1/12  Call with integers for offset and length. (goes with double-precision changes)
 *
 */

#include <Python.h> /* make sure Python.h is first */
#include <stdlib.h>
#include "ferret.h"
#include "FerMem.h"

void FORTRAN(c_substr)(char **in_ptr, int *offset, int *length, char **out_ptr)
{
   char* tmp;
   char* tmp2;
   int i, int_length, int_offset;

   if ( (*out_ptr != NULL) && (*out_ptr != STRING_MISSING_VALUE) )
      FerMem_Free(*out_ptr, __FILE__, __LINE__);
   if ( *in_ptr == NULL ) {
      /* undefined string given, so return an undefined string */
      *out_ptr = NULL;
      return;
   }
   if ( *in_ptr == STRING_MISSING_VALUE ) {
      /* missing string given, so return an missing string */
      *out_ptr = STRING_MISSING_VALUE;
      return;
   }

   int_length = *length;
   int_offset = *offset - 1;

   *out_ptr = (char *) FerMem_Malloc(sizeof(char) * (int_length + 1), __FILE__, __LINE__);
   if ( *out_ptr == NULL )
      abort();

   tmp2 = *in_ptr;
   for (i = 0; (i < int_offset) && (*tmp2 != '\0'); i++)
      tmp2++;
   tmp = *out_ptr;
   for (i = 0; (i < int_length) && (*tmp2 != '\0'); i++) {
      *tmp = *tmp2;
      tmp++;
      tmp2++;
   }
   *tmp = '\0';
}
