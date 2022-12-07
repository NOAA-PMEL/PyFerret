*
* appende.F
*
* This function creates a new variable which is the first variable
* with the second variable appended on the ensemble axis.
*

* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

*
* In this subroutine we provide information about
* the function.  The user configurable information
* consists of the following:
*
* descr              Text description of the function
*
* num_args           Required number of arguments
*
* axis_inheritance   Type of axis for the result
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )
*
*
* For each argument we provide the following information:
*
* name               Text name for an argument
*
* unit               Text units for an argument
*
* desc               Text description of an argument
*
* axis_influence     Are this argument's axes the same as the result grid?
*                       ( YES, NO )
*
* axis_extend       How much does Ferret need to extend arg limits relative to result
*


      SUBROUTINE appende_init(id)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg



* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .        'appends VAR to ENS along the ensemble axis')
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                CUSTOM,          IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'ENS')
      CALL ef_set_arg_desc(id, arg, 'Initial variable')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'VAR')
      CALL ef_set_arg_desc(id, arg, 'Variable to append')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, NO, YES)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we provide information about the custom axis.
*

      SUBROUTINE appende_custom_axes(id)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      REAL my_lo, my_hi, my_delta
      CHARACTER*20 my_units

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      IF ( (arg_lo_ss(E_AXIS,ARG1) .EQ. ef_unspecified_int4) .AND.
     .     (arg_hi_ss(E_AXIS,ARG1) .EQ. ef_unspecified_int4) ) THEN
         my_lo = 1.0
         my_hi = 1.0
      ELSE
         my_lo = arg_lo_ss(E_AXIS,ARG1)
         my_hi = arg_hi_ss(E_AXIS,ARG1)
      ENDIF
      my_hi = my_hi + 
     .        arg_hi_ss(E_AXIS,ARG2) - arg_lo_ss(E_AXIS,ARG2) + 1.0
      my_delta = 1.0
      my_units = ' '

      CALL ef_set_custom_axis(id, E_AXIS, my_lo, my_hi,
     .                        my_delta, my_units, NO)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE appende_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

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
* information for each variable's axes.

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,  j,  k,  l,  m,  n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*     Initialize everything to undefined so nothing is missed
      DO n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
      DO m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
      DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
      DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
      DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         result(i,j,k,l,m,n) = bad_flag_result
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

*     Start by copying the first variable
*     (Ensemble axis must be outer-most loop)
      m = res_lo_ss(E_AXIS)
      DO 510 m1 = arg_lo_ss(E_AXIS,ARG1), arg_hi_ss(E_AXIS,ARG1)

         n1 = arg_lo_ss(F_AXIS,ARG1)
         DO 610 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         l1 = arg_lo_ss(T_AXIS,ARG1)
         DO 410 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         k1 = arg_lo_ss(Z_AXIS,ARG1)
         DO 310 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 210 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         i1 = arg_lo_ss(X_AXIS,ARG1)
         DO 110 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            IF ( arg_1(i1,j1,k1,l1,m1,n1) .NE. bad_flag(ARG1) ) THEN

               result(i,j,k,l,m,n) = arg_1(i1,j1,k1,l1,m1,n1)

            END IF

            i1 = i1 + arg_incr(X_AXIS,ARG1)
 110     CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 210     CONTINUE

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
 310     CONTINUE

            l1 = l1 + arg_incr(T_AXIS,ARG1)
 410     CONTINUE

            n1 = n1 + arg_incr(F_AXIS,ARG1)
 610     CONTINUE

         m = m + 1
 510  CONTINUE

*     Now copy the second variable starting on the ensemble axis
*     where the first variable left off (value of m not reset)

      DO 520 m2 = arg_lo_ss(E_AXIS,ARG2), arg_hi_ss(E_AXIS,ARG2)

         n2 = arg_lo_ss(F_AXIS,ARG2)
         DO 620 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         l2 = arg_lo_ss(T_AXIS,ARG2)
         DO 420 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         k2 = arg_lo_ss(Z_AXIS,ARG2)
         DO 320 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

         j2 = arg_lo_ss(Y_AXIS,ARG2)
         DO 220 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         i2 = arg_lo_ss(X_AXIS,ARG2)
         DO 120 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            IF ( arg_2(i2,j2,k2,l2,m2,n2) .NE. bad_flag(ARG2) ) THEN

               result(i,j,k,l,m,n) = arg_2(i2,j2,k2,l2,m2,n2)

            END IF

            i2 = i2 + arg_incr(X_AXIS,ARG2)
 120     CONTINUE

            j2 = j2 + arg_incr(Y_AXIS,ARG2)
 220     CONTINUE

            k2 = k2 + arg_incr(Z_AXIS,ARG2)
 320     CONTINUE

            l2 = l2 + arg_incr(T_AXIS,ARG2)
 420     CONTINUE

            n2 = n2 + arg_incr(F_AXIS,ARG2)
 620     CONTINUE

         m = m + 1
 520  CONTINUE

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END
