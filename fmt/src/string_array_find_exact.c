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
  06/04 *ywei* Created to find a string in a string array fast.
               The matching method is exactly equal.
 */
#include <stdio.h>
#include "string_array.h"

void string_array_find_exact_(    int *string_array_header,
			          char *test_string,
                                  int *test_len,
                                  int *result_array,
                                  int *result_array_size,
                                  int *num_indices
                             )
{
   int i,j=0;
   int true_test_len, true_model_len, array_size, 
       string_size, hash_value, result_array_size1;
   SA_Head * head;
   List_Node *bucket, *p;
   char * model_string;
   int match=0;

 
   if(*string_array_header){
      head = *((SA_Head**)string_array_header);
      array_size = head->array_size;
      string_size = head->string_size;
 
      tm_get_strlen_(&true_test_len, test_len, test_string);
 
 
      hash_value = string_array_hash(test_string, true_test_len, 0, array_size);
      

      if(true_test_len ==0){
	 result_array_size1 = 5;
      }
      else {
	 result_array_size1 = *result_array_size;
      } 

      bucket = head->hash_table[hash_value];

      for(p=bucket; p; p=p->next) {
	  model_string=&(head->string_array[(p->index-1)*string_size]);
          string_array_get_strlen_(string_array_header, &(p->index), &true_model_len);

          match = 0;

          if(true_model_len == true_test_len){
	     match = 1;
	     for( i=0; i<true_model_len; i++){
		 if(test_string[i]!=model_string[i]){
		    match = 0;
                    break;
		 }
	     }
	  }
	  

          if(match==1){
	      if(j<result_array_size1){
		 result_array[j]=p->index;
                 j++;
	      }
              else{
                 break;
	      }
	  }
      }
   }
   else{
       printf("\nString array not initialized yet!");
   }
   *num_indices = j;
}


