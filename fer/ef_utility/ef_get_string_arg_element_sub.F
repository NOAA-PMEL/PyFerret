* ef_get_string_arg_element_sub.F
*
*  acm 12/03  from ef_get_arg_string
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
* This returns an element of the string array which was passed as an argument
* to the function.  

* Inside this subroutine arg is a 4-D real*8 array of pointers, and
* we can call GET_STRING_ELEMENT to get a string element.

      SUBROUTINE ef_get_string_arg_element_sub (lox, hix, loy, hiy,
     .                     loz, hiz, lot, hit, arg, i, j, k, l, 
     .			   max_line_len, slen, text )

      INTEGER i, j, k, l, lox, hix, loy, hiy, loz, hiz, lot, hit, 
     .  slen, max_line_len
      CHARACTER*(*) text
      REAL*8 arg(lox:hix, loy:hiy, loz:hiz, lot:hit)

      CHARACTER	GET_STRING_ELEMENT*512
    

      text = GET_STRING_ELEMENT (lox, hix, loy, hiy, loz, hiz, lot, hit,
     .			         arg, i, j, k, l, max_line_len, slen)

      RETURN

      END
