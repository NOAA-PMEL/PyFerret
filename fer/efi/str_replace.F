*
* str_replace.F
*
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
* Ansley Manke
* 8/2016
*
* Returns string with all instnces of one character replaced by a given single character.
*


      SUBROUTINE str_replace_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*120 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns strings with all instances of a character replaced')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_result_type(id, STRING_RETURN)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'INSTRING')
      CALL ef_set_arg_desc(id, arg, 'String variable to edit')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)
      CALL ef_set_arg_type(id, arg, STRING_ARG)
      

      arg = 2
      CALL ef_set_arg_name(id, arg, 'S1')
      CALL ef_set_arg_desc(id, arg, 'Single character to replace')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)
      
      arg = 3
      CALL ef_set_arg_name(id, arg, 'S2')
      CALL ef_set_arg_desc(id, arg, 'Single character to insert')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)


      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE str_replace_compute(id, arg_1, arg_2, arg_3, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'


      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy,
     .           mem1loz:mem1hiz, mem1lot:mem1hit,
     .           mem1loe:mem1hie, mem1lof:mem1hif)

      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .           mem2loz:mem1hiz, mem1lot:mem1hit,
     .           mem1loe:mem1hie, mem1lof:mem1hif)

      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy,
     .           mem3loz:mem3hiz, mem3lot:mem3hit,
     .           mem3loe:mem3hie, mem3lof:mem3hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* Ignore the Y/Z/T/E/F dimensions in the work arrays since
* they are not used and Fortran is column major

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable''s axes. 

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      
      CHARACTER*2048 buff
      CHARACTER*1    c_remove, c_replace
      CHARACTER*100  err_msg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER slen, ii
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      DO ii = X_AXIS, F_AXIS
         IF (arg_hi_ss(ii,2) - arg_lo_ss(ii,2) .GT. 1) THEN
            WRITE (err_msg,*) 'Argument 2 is a single-character string constant'
	    GOTO 999
         ENDIF
         IF (arg_hi_ss(ii,3) - arg_lo_ss(ii,3) .GT. 1) THEN
            WRITE (err_msg,*) 'Argument 3 is a single-character string constant'
	    GOTO 999
         ENDIF
      ENDDO

      CALL EF_GET_STRING_ARG_ELEMENT_6D(id, ARG2, arg_2,
     .  mem2lox, mem2lox, mem2lox, mem2lox, mem2lox, mem2lox, slen, c_remove)
      IF (slen .NE. 1) THEN
         WRITE (err_msg,*) 'Argument 2 is a single-character string constant'
         GOTO 999
      ENDIF

      CALL EF_GET_STRING_ARG_ELEMENT_6D(id, ARG3, arg_3,
     .  mem3lox, mem3lox, mem3lox, mem3lox, mem3lox, mem3lox, slen, c_replace)
      IF (slen .NE. 1) THEN
         WRITE (err_msg,*) 'Argument 3 is a single-character string constant'
         GOTO 999
      ENDIF

      n1 = arg_lo_ss(F_AXIS, ARG1)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m1 = arg_lo_ss(E_AXIS, ARG1)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      l1 = arg_lo_ss(T_AXIS, ARG1)
      DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         k1 = arg_lo_ss(Z_AXIS, ARG1)
         DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            j1 = arg_lo_ss(Y_AXIS, ARG1)
            DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

               i1 = arg_lo_ss(X_AXIS, ARG1)
               DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

                  CALL EF_GET_STRING_ARG_ELEMENT_6D(id, ARG1, arg_1,
     .                    i1, j1, k1, l1, m1, n1, slen, buff)
                  IF (slen .LE. 0) THEN
                     CALL EF_PUT_STRING(buff, slen, result(i,j,k,l,m,n))
                  ELSE
                     CALL REPLACE_LETTER(buff, slen, c_remove, c_replace)
                     CALL EF_PUT_STRING(buff, slen, result(i,j,k,l,m,n))
                  ENDIF

                  i1 = i1 + arg_incr(X_AXIS,ARG1)
 100           CONTINUE

               j1 = j1 + arg_incr(Y_AXIS, ARG1)
 200        CONTINUE

            k1 = k1 + arg_incr(Z_AXIS, ARG1)
 300     CONTINUE

         l1 = l1 + arg_incr(T_AXIS, ARG1)
 400  CONTINUE

         m1 = m1 + arg_incr(E_AXIS, ARG1)
 500  CONTINUE

         n1 = n1 + arg_incr(F_AXIS, ARG1)
 600  CONTINUE

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
 999  CALL ef_bail_out (id, err_msg)

      END

      SUBROUTINE REPLACE_LETTER(buff, slen, c_remove, c_replace)

      INTEGER slen
      CHARACTER*1 buff(*), c_remove, c_replace

      INTEGER i

      DO i = 1, slen
         IF (buff(i) .EQ. c_remove) buff(i) = c_replace
      ENDDO

      RETURN
      END


