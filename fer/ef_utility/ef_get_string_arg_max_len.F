      SUBROUTINE EF_GET_STRING_ARG_MAX_LEN(id, iarg, str_arg, smax)
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
*  acm 12/03  
*
* This returns the maximum length of the string elements of a string array
* which was passed as an argument to the external function.
*
* V691+*acm* 8/14  Fix ticket 2190: long strings were being cut off.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'EF_Util.parm'

      INTEGER id, iarg, smax
      REAL str_arg(*)

      INTEGER max_line_len, mr, slen,
     .        i, j, k, l, m, n, 
     .        lox, hix, loy, hiy, loz, hiz,
     .        lot, hit, loe, hie, lof, hif
      INTEGER mr_list(EF_MAX_ARGS)
      CHARACTER buff*64
      PARAMETER (max_line_len = 2048)
      
      INTEGER arg_lo_ss(nferdims,EF_MAX_ARGS),
     .        arg_hi_ss(nferdims,EF_MAX_ARGS),
     .        arg_incr (nferdims,EF_MAX_ARGS)

      CALL EF_GET_ARG_SUBSCRIPTS(id, arg_lo_ss, arg_hi_ss, arg_incr)
 
* The memory resident list was cached by Ferret and must be retrieved
* to be accessible to the external function.

      CALL EF_GET_MR_LIST(mr_list)
      mr = mr_list(iarg)

      IF ( mr .EQ. 0 ) THEN
* Do not call during custom axis creation time
          buff = 
     .     'Use ef_get_string_arg_element_len in the compute phase'
          GOTO 5000

* Check the variable type.

      ELSE IF (mr_type(mr) .NE. ptype_string) THEN
         WRITE (buff,4000) iarg
 4000    FORMAT ('Argument',I3,' is not a string')
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
         loe = arg_lo_ss(E_AXIS,iarg)
         hie = arg_hi_ss(E_AXIS,iarg)
         lof = arg_lo_ss(F_AXIS,iarg)
         hif = arg_hi_ss(F_AXIS,iarg)


* Inside ef_get_string_arg_elem_len_sub, str_arg is an array of pointers (cast as REAL*8),
* and we can call GET_STRING_ELEMENT to get the length of a string element. 

          smax = 0
          DO n = lof, hif
          DO m = loe, hie
          DO l = lot, hit
          DO k = loz, hiz
          DO j = loy, hiy
          DO i = lox, hix
             CALL EF_GET_STRING_ARG_ELEM_LEN_SUB(lox, hix, loy, hiy,
     .                   loz, hiz, lot, hit, loe, hie, lof, hif,
     .                   str_arg, i, j, k, l, m, n, max_line_len, slen)
             smax = MAX(slen, smax)
          ENDDO
          ENDDO
          ENDDO
          ENDDO
          ENDDO
          ENDDO

      ENDIF

      RETURN

* return an error message     
 5000 CALL EF_BAIL_OUT(id, buff)
      STOP 'EF_BAIL_OUT returned in EF_GET_STRING_ARG_MAX_LEN'

      END
