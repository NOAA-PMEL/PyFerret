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
                 for better performance. This function is to modify a variable
                 in an array and keep the two lists updated.
 */
#include <stdio.h>
#include "deleted_list.h"

void deleted_list_modify_(void *deleted_list_header,
                           int  *index,
                           int  *new_value){
   DLHead * head;
   DL_Node * p;
   int old_value;

   head = *((DLHead**) deleted_list_header);

   if(head){
      old_value = head->int_array[*index-1];
      head->int_array[*index-1]=*new_value;
      p = head->ptr_table[*index-1];

      if(old_value==head->deleted_value
         && *new_value!=head->deleted_value){
          if(p->prev){
	     p->prev->next = p->next;
          }
          if(p->next){
	     p->next->prev = p->prev;
          }
          if(head->deleted_list_head == p){
	     head->deleted_list_head = p->next;
          }

          p->prev = NULL;
          p->next = head->undel_list_head;
          head->undel_list_head = p;
          if(p->next){
 	      p->next->prev = p;
          }
      } 
      else if(old_value!=head->deleted_value
	      && *new_value==head->deleted_value){
          if(p->prev){
	     p->prev->next = p->next;
          }
          if(p->next){
	     p->next->prev = p->prev;
          }
          if(head->undel_list_head == p){
	     head->undel_list_head = p->next;
          }

          p->prev = NULL;
          p->next = head->deleted_list_head;
          head->deleted_list_head = p;
          if(p->next){
 	      p->next->prev = p;
          }
      }
   }
}

