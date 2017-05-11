*
* dot_f.F
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
* May 2017 Ansley Manke
*
* compute the dot product of two arguments along a shared axis
*




      SUBROUTINE dot_f_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_set_desc(id, 'Compute the inner product of two variables along F-axis')
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'VAR1')
      CALL ef_set_arg_desc(id, arg, 'Variable 1')
      CALL ef_set_axis_influence_6d(id, arg,
     .                                YES, YES, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'VAR2')
      CALL ef_set_arg_desc(id, arg, 'Variable 2')
      CALL ef_set_axis_influence_6d(id, arg,
     .                                YES, YES, YES, YES, YES, YES)

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE dot_f_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable''s axes. 

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1, i2, j2, k2, l2, m2, n2
      CHARACTER*100 errtxt

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* check to make sure both variables share the common axis
      IF ( (arg_lo_ss(F_AXIS, ARG1) .EQ. ef_unspecified_int4) .OR. 
     .     (arg_hi_ss(F_AXIS, ARG1) .EQ. ef_unspecified_int4) .OR. 
     .     (arg_lo_ss(F_AXIS, ARG2) .EQ. ef_unspecified_int4) .OR. 
     .     (arg_hi_ss(F_AXIS, ARG2) .EQ. ef_unspecified_int4) 
     .      ) THEN
         WRITE(errtxt,*) 'F axis of arguments 1 and 2 cannot be a normal axis'
         GOTO 999
      ENDIF
      IF ( arg_hi_ss(F_AXIS, ARG1) - arg_lo_ss(F_AXIS, ARG1) .NE. 
     .     arg_hi_ss(F_AXIS, ARG2) - arg_lo_ss(F_AXIS, ARG2) ) THEN
         WRITE(errtxt,*) 'F axis of Arguments 1 and 2 must be the same length'
         GOTO 999
      ENDIF

* Initialize the result, F90 initialize whole array

      result = bad_flag_result  

* Compute the dot product

      m1 = arg_lo_ss(E_AXIS,ARG1)
      m2 = arg_lo_ss(E_AXIS,ARG2)
      DO 600 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

       l1 = arg_lo_ss(T_AXIS,ARG1)
       l2 = arg_lo_ss(T_AXIS,ARG2)
       DO 500 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

        k1 = arg_lo_ss(Z_AXIS,ARG1)
        k2 = arg_lo_ss(Z_AXIS,ARG2)
        DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         DO 300 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

          i1 = arg_lo_ss(X_AXIS,ARG1)
          i2 = arg_lo_ss(X_AXIS,ARG2)
          DO 200 i = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)

           n  = res_lo_ss(F_AXIS)
           n2 = arg_lo_ss(F_AXIS,ARG2)
           DO 100 n1 = arg_lo_ss(F_AXIS,ARG1), arg_hi_ss(F_AXIS,ARG1)

            IF ( arg_1(i1,j1,k1,l1,m1,n1) .NE. bad_flag(ARG1) .AND. 
     .	         arg_2(i2,j2,k2,l2,m2,n2) .NE. bad_flag(ARG2) ) THEN

               IF (result(i,j,k,l,m,n) .EQ. bad_flag_result) THEN
	          result(i,j,k,l,m,n) =  
     .	            arg_1(i1,j1,k1,l1,m1,n1)* arg_2(i2,j2,k2,l2,m2,n2)

               ELSE

	          result(i,j,k,l,m,n) =  result(i,j,k,l,m,n) + 
     .	            arg_1(i1,j1,k1,l1,m1,n1)* arg_2(i2,j2,k2,l2,m2,n2)

               ENDIF

            ENDIF

           n2 = n2 + arg_incr(F_AXIS,ARG2)
 100       CONTINUE

            i1 = i1 + arg_incr(X_AXIS,ARG1)
            i2 = i2 + arg_incr(X_AXIS,ARG2)
 200       CONTINUE

           j1 = j1 + arg_incr(Y_AXIS,ARG1)
           j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300      CONTINUE

          k1 = k1 + arg_incr(Z_AXIS,ARG1)
          k2 = k2 + arg_incr(Z_AXIS,ARG2)
 400     CONTINUE

         l1 = l1 + arg_incr(T_AXIS,ARG1)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
 500    CONTINUE

        m1 = m1 + arg_incr(E_AXIS,ARG1)
        m2 = m2 + arg_incr(E_AXIS,ARG2)
 600   CONTINUE


      RETURN

 999  CALL EF_BAIL_OUT(id, errtxt)

      RETURN 
      END
