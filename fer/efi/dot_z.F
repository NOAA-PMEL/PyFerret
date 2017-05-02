*
* dot_z.F
*
* compute the dot product of two arguments along a shared axis
*




      SUBROUTINE dot_z_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_set_desc(id, 'Compute the inner product of two variables along Z-axis')
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                NORMAL, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
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
      SUBROUTINE dot_z_compute(id, arg_1, arg_2, result)

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
      IF ( (arg_lo_ss(Z_AXIS, ARG1) .EQ. ef_unspecified_int4) .OR. 
     .     (arg_hi_ss(Z_AXIS, ARG1) .EQ. ef_unspecified_int4) .OR. 
     .     (arg_lo_ss(Z_AXIS, ARG2) .EQ. ef_unspecified_int4) .OR. 
     .     (arg_hi_ss(Z_AXIS, ARG2) .EQ. ef_unspecified_int4) 
     .      ) THEN
         WRITE(errtxt,*) 'Z axis of arguments 1 and 2 cannot be a normal axis'
         GOTO 999
      ENDIF
      IF ( arg_hi_ss(Z_AXIS, ARG1) - arg_lo_ss(Z_AXIS, ARG1) .NE. 
     .     arg_hi_ss(Z_AXIS, ARG2) - arg_lo_ss(Z_AXIS, ARG2) ) THEN
         WRITE(errtxt,*) 'Z axis of Arguments 1 and 2 must be the same length'
         GOTO 999
      ENDIF

* Initialize the result

      DO 90 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
      DO 90 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
      DO 90 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
      DO 90 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
      DO 90 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
      DO 90 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         result(i,j,k,l,m,n) = bad_flag_result
  90  CONTINUE

      n1 = arg_lo_ss(F_AXIS,ARG1)
      n2 = arg_lo_ss(F_AXIS,ARG2)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

       m1 = arg_lo_ss(E_AXIS,ARG1)
       m2 = arg_lo_ss(E_AXIS,ARG2)
       DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

        l1 = arg_lo_ss(T_AXIS,ARG1)
        l2 = arg_lo_ss(T_AXIS,ARG2)
        DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         DO 300 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

          i1 = arg_lo_ss(X_AXIS,ARG1)
          i2 = arg_lo_ss(X_AXIS,ARG2)
          DO 200 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

	   k  = res_lo_ss(Z_AXIS)
           k2 = arg_lo_ss(Z_AXIS,ARG2)
           DO 100 k1 = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)

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

           k2 = k2 + arg_incr(Z_AXIS,ARG2)
 100       CONTINUE

           i1 = i1 + arg_incr(X_AXIS,ARG1)
           i2 = i2 + arg_incr(X_AXIS,ARG2)
 200      CONTINUE

          j1 = j1 + arg_incr(Y_AXIS,ARG1)
          j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300     CONTINUE

         l1 = l1 + arg_incr(T_AXIS,ARG1)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
 400    CONTINUE

        m1 = m1 + arg_incr(E_AXIS,ARG1)
        m2 = m2 + arg_incr(E_AXIS,ARG2)
 500   CONTINUE

       n1 = n1 + arg_incr(F_AXIS,ARG1)
       n2 = n2 + arg_incr(F_AXIS,ARG2)
 600  CONTINUE

      RETURN

 999  CALL EF_BAIL_OUT(id, errtxt)

      RETURN 
      END
