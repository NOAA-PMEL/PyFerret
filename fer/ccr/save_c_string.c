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
 *   Allocate dynamic memory and save a string returning the pointer to it
 */

#include <Python.h> /* make sure Python.h is first */
#include "ferret.h"
#include "FerMem.h"

void FORTRAN(save_c_string)(char *string, int *inlen, char ***fer_ptr, int *offset, int *stat)
{
   int i;
   char* ptr;
   char** each_str_ptr;

   ptr = (char *) FerMem_Malloc(sizeof(char) * (*inlen + 1), __FILE__, __LINE__);
   if ( ptr != NULL ) {
      for (i=0; i<*inlen; i++)
         ptr[i] = string[i];
      ptr[*inlen] = '\0';
      
      /* save the pointer to the string */
      each_str_ptr = *fer_ptr;   /* holds pointer to the first string */
      each_str_ptr += *offset * 8/sizeof(char**); /* point to the desired string */ 
      if ( (*each_str_ptr != NULL) && (*each_str_ptr != STRING_MISSING_VALUE) )
         FerMem_Free( *each_str_ptr, __FILE__, __LINE__ );
      *each_str_ptr = ptr;

      *stat = 0;
   }
   else
      *stat = 1;
}
