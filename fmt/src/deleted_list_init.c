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
   06/04 *ywei* -Created to keep two lists  deleted list and undeleted list)
                 for better performance. This function is to initialize the 
                 data structure.
 */
/* *acm   9/06 v600 - add stdlib.h wherever there is stdio.h for altix build*/ 
#include <stddef.h>  /* size_t, ptrdiff_t; gfortran on linux rh5*/
# include <stdio.h>
#include <stdlib.h>
#include "deleted_list.h"

void deleted_list_init_(void *deleted_list_header,
                         int *int_array,
			 int *int_array_size,
                         int *deleted_value 
                        )
{
   int i,j;
   int array_size;
   DLHead * head;
   DL_Node * p;

   head = (DLHead*)malloc(sizeof(DLHead));
   *((DLHead**)deleted_list_header) = head;
   head->int_array = int_array;
   head->array_size = *int_array_size;
   head->deleted_value = *deleted_value;
   array_size = head->array_size;

   head->ptr_table = (DL_Node**)malloc(array_size*sizeof(DL_Node*));
   memset((void *)head->ptr_table, 0, array_size*sizeof(DL_Node*));
   head->deleted_list_head = NULL;
   head->undel_list_head = NULL;

   for(j=array_size;j>=1;j--) {
       head->ptr_table[j-1] = (DL_Node*)malloc(sizeof(DL_Node));
       p = head->ptr_table[j-1];
       p->index = j;

       if(head->int_array[j-1]==head->deleted_value){
          p->prev = NULL;
          p->next = head->deleted_list_head;
          head->deleted_list_head = p;
          if(p->next){
 	     p->next->prev = p;
          }
       }
       else{
          p->prev = NULL;
          p->next = head->undel_list_head;
          head->undel_list_head = p;
          if(p->next){
 	     p->next->prev = p;
          }
       }
   }
}

