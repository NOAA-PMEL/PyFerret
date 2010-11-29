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
*  ywei: 05/04 created to speed up uppercase string matching
*/
#include <stdio.h>

void str_case_blind_compare_sub_(char* test_name,
                            int * len_test,
                            char* model_name, 
                            int * len_model, 
                            int * result)
{
     int i, ltest=*len_test, lmod=*len_model;
     char c1, c2;

     *result=0;
     if(ltest<lmod){
         for(i=0; i<ltest;i++)
         { 
            c1 = test_name[i];
            c2 = model_name[i];

            if(c1!=c2) {
	       if(c1>='a' && c1<='z'){
 		  c1&=0xDF;
	       }
               if(c2>='a' && c2<='z'){
 		  c2&=0xDF;
	       }
               if(c1<c2){
                 *result=-1;
	         return;
	       }
               else if(c1>c2){
                 *result=1;
                 return;
	       }
	    }                
         }

         for(i=ltest;i<lmod;i++){
	     if(model_name[i]!=' '){
	        *result=-1;
                return;
	     }
	 }
     }
     else{
         for(i=0; i<lmod;i++)
         { 
            c1 = test_name[i];
            c2 = model_name[i];

            if(c1!=c2) {
	       if(c1>='a' && c1<='z'){
 		  c1&=0xDF;
	       }
               if(c2>='a' && c2<='z'){
 		  c2&=0xDF;
	       }
               if(c1<c2){
                 *result=-1;
                 return;
               }
               else if(c1>c2){
                 *result=1;
		 return;
	       }
	    }                
         }

         for(i=lmod;i<ltest;i++){
	     if(test_name[i]!=' '){
	        *result=1;
                return;
	     }
	 }
     }
     return;
}
