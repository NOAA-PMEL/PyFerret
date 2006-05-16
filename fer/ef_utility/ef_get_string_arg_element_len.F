      SUBROUTINE ef_get_string_arg_element_len 
     .            (id, iarg, arg, i,j,k,l, slen)
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
*
*  acm 12/03  
*
* This returns the length of the I,J,K,L-th element of the string array which 
* was passed as an argument to the external function.
*

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'EF_Util.parm'

      INTEGER id, iarg, slen, i, j, k, l
      REAL arg(*)

      INTEGER max_line_len, mr, lox, hix, loy, hiy, loz, hiz, lot, hit
      INTEGER mr_list(EF_MAX_ARGS)
      CHARACTER buff*32
      PARAMETER (max_line_len = 512)
      
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
 
* The memory resident list was cached by Ferret and must be retrieved
* to be accessible to the external function.

      CALL EF_GET_MR_LIST(mr_list)
      mr = mr_list(iarg)
      
      IF ( mr .EQ. 0 ) THEN
* Don't call during custom axis creation time
          buff = 
     .     'Use ef_get_string_arg_element_len in the compute phase'
          GOTO 5000

* Here's how you check the variable type.
* (this should never happen as checks against it are made previously) 
        ELSE IF (mr_type(mr) .NE. ptype_string) THEN
           WRITE (buff,4000) iarg
 4000      FORMAT ('Argument',I3,' is not a string')
           GOTO 5000
        ELSE

           lox = arg_lo_ss(X_AXIS,iarg)
           hix = arg_hi_ss(X_AXIS,iarg)
           loy = arg_lo_ss(Y_AXIS,iarg)
           hiy = arg_hi_ss(Y_AXIS,iarg)
           loz = arg_lo_ss(Z_AXIS,iarg)
           hiz = arg_hi_ss(Z_AXIS,iarg)
           lot = arg_lo_ss(T_AXIS,iarg)
           hit = arg_hi_ss(T_AXIS,iarg)


* Inside ef_get_string_arg_elem_len_sub, arg is a real*8 array of pointers, and
* we can call GET_STRING_ELEMENT to get the length of a string element.

           CALL ef_get_string_arg_elem_len_sub (
     .                              lox, hix, loy, hiy,
     .                              loz, hiz, lot, hit,
     .			            arg, i, j, k, l, 
     .			            max_line_len, slen)

      ENDIF

      RETURN

* return an error message     
 5000 CALL EF_BAIL_OUT(id, buff)

      END
