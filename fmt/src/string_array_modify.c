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
   06/04 *ywei* -Created to modify a string inside a string array.
                 This function should be used to keep the hashtable
                 up to date.
 */
#include <stdio.h>
#include "string_array.h"

void string_array_modify_(   int  *string_array_header,
                              int  *index,
                              char *new_string,
                              int  *new_string_size){

   int true_old_str_len, true_new_str_len,
       array_size, string_size, old_hash_value,
       new_hash_value;
   SA_Head * head;
   char * old_string;
   int i;
   List_Node * p;
 FILE *fp;

   if(*string_array_header ==1 ) {
      head = (SA_Head*)string_array_header;
      array_size = head->array_size;
      string_size = head->string_size;

      old_string = &(head->string_array[(*index-1)*string_size]);
      string_array_get_strlen_(string_array_header,index, &true_old_str_len);
      old_hash_value = string_array_hash(old_string, true_old_str_len, 0, array_size);

      tm_get_strlen_(&true_new_str_len, new_string_size, new_string);
      if(true_new_str_len>string_size)
	true_new_str_len = string_size;
      new_hash_value = string_array_hash(new_string, true_new_str_len, 0, array_size);

      if(old_hash_value != new_hash_value){

	  p = head->ptr_array[*index-1];
          if(head->hash_table[old_hash_value]==p)
	     head->hash_table[old_hash_value] = p->next;
          if(p->prev)
	     p->prev->next = p->next;
          if(p->next)
	     p->next->prev = p->prev;

          p->prev = NULL;
          p->next = head->hash_table[new_hash_value];
          head->hash_table[new_hash_value] = p;
          if(p->next)
	     p->next->prev = p;             
      }

      for(i=0;i<true_new_str_len;i++){
	 old_string[i] = new_string[i];
      }
      for(i=true_new_str_len;i<string_size;i++){
	 old_string[i] = ' ';
      }
      head->strlen_array[*index-1]=true_new_str_len;
   }
}

