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
/*
   06/04 *ywei* -Created to clean the allocated memory used by string_array
                 functiions
    4/06 *kob*  change type of argument to double, for 64-bit build
 */

#include <stdlib.h>
#include <string.h>
#include "fmtprotos.h"
#include "string_array.h"
#include "FerMem.h"

void FORTRAN(string_array_clear)(void **string_array_header)
{
    SA_Head *head; 
    int j;

    if ( *string_array_header != NULL ) {
       head = *string_array_header;
       for (j = 0; j < head->array_size; j++) {
	  FerMem_Free(head->ptr_array[j], __FILE__, __LINE__);
	  head->ptr_array[j] = NULL;
       }
       FerMem_Free(head->ptr_array, __FILE__, __LINE__);
       FerMem_Free(head->strlen_array, __FILE__, __LINE__);
       FerMem_Free(head->hash_table, __FILE__, __LINE__);
       memset(head, 0, sizeof(SA_Head));
       FerMem_Free(head, __FILE__, __LINE__);
       *string_array_header = NULL;
    }
}

