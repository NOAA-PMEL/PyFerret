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
  06/04 *ywei* -Created to initialize data structure for a string_array
                function group.
    4/06 *kob*  change type of 1st argument to double, for 64-bit build
 */
/* *acm   9/06 v600 - add stdlib.h wherever there is stdio.h for altix build*/ 
#include <stdio.h>
#include <stdlib.h>
#include "string_array.h"

void string_array_init_( double *string_array_header,
                         int  *array_size,
                         int  *string_size,
                         char *string_array
                        )
{
   int i,j;
   int true_len, hash_value;
   SA_Head * head;
   List_Node * p;
   
   head = (SA_Head*)malloc(sizeof(SA_Head));
   *((SA_Head**)string_array_header) = head;
   head->array_size = *array_size;
   head->string_size = *string_size;
   head->string_array = string_array;

   head->ptr_array = (List_Node**)malloc(head->array_size*sizeof(List_Node*));

   head->hash_table = (List_Node**)malloc(head->array_size*sizeof(List_Node*));
   memset((void*)head->hash_table, 0, head->array_size*sizeof(List_Node*));

   head->strlen_array = (int*)malloc(head->array_size*sizeof(int));

   for(j=head->array_size;j>=1;j--) {
       tm_get_strlen_(&true_len, &(head->string_size),
		     &(head->string_array[(j-1)*head->string_size]));
       head->strlen_array[j-1]=true_len;

       hash_value = string_array_hash(&(head->string_array[(j-1)*head->string_size]),
                       true_len, 0, head->array_size);

       head->ptr_array[j-1] = (List_Node*)malloc(sizeof(List_Node));
       p = head->ptr_array[j-1];
       p->index = j;
       p->prev = NULL;
       p->next = head->hash_table[hash_value];
       head->hash_table[hash_value] = p;

       if(p->next)
	  p->next->prev = p;
   }

}

