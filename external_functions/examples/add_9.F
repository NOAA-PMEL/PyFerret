*
* add_9.F
*
* Jonathan Callahan
* Feb 19th 1998
*
* Returns the sum of nine arguments.
*


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


      SUBROUTINE add_9_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,'adds 9 arguments' )

      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

* All arguments should be on the same grid

      arg = 1
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 3
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 4
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 5
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 6
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 7
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 8
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 9
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE add_9_compute(id, arg_1, arg_2, arg_3, arg_4, arg_5,
     .     arg_6, arg_7, arg_8, arg_9, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .     mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .     mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .     mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, 
     .     mem4loz:mem4hiz, mem4lot:mem4hit)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, 
     .     mem5loz:mem5hiz, mem5lot:mem5hit)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, 
     .     mem6loz:mem6hiz, mem6lot:mem6hit)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, 
     .     mem7loz:mem7hiz, mem7lot:mem7hit)
      REAL arg_8(mem8lox:mem8hix, mem8loy:mem8hiy, 
     .     mem8loz:mem8hiz, mem8lot:mem8hit)
      REAL arg_9(mem9lox:mem9hix, mem9loy:mem9hiy, 
     .     mem9loz:mem9hiz, mem9lot:mem9hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2
      INTEGER i3, j3, k3, l3
      INTEGER i4, j4, k4, l4
      INTEGER i5, j5, k5, l5
      INTEGER i6, j6, k6, l6
      INTEGER i7, j7, k7, l7
      INTEGER i8, j8, k8, l8
      INTEGER i9, j9, k9, l9

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)
      
      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      i3 = arg_lo_ss(X_AXIS,ARG3)
      i4 = arg_lo_ss(X_AXIS,ARG4)
      i5 = arg_lo_ss(X_AXIS,ARG5)
      i6 = arg_lo_ss(X_AXIS,ARG6)
      i7 = arg_lo_ss(X_AXIS,ARG7)
      i8 = arg_lo_ss(X_AXIS,ARG8)
      i9 = arg_lo_ss(X_AXIS,ARG9)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         j3 = arg_lo_ss(Y_AXIS,ARG3)
         j4 = arg_lo_ss(Y_AXIS,ARG4)
         j5 = arg_lo_ss(Y_AXIS,ARG5)
         j6 = arg_lo_ss(Y_AXIS,ARG6)
         j7 = arg_lo_ss(Y_AXIS,ARG7)
         j8 = arg_lo_ss(Y_AXIS,ARG8)
         j9 = arg_lo_ss(Y_AXIS,ARG9)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            k2 = arg_lo_ss(Z_AXIS,ARG2)
            k3 = arg_lo_ss(Z_AXIS,ARG3)
            k4 = arg_lo_ss(Z_AXIS,ARG4)
            k5 = arg_lo_ss(Z_AXIS,ARG5)
            k6 = arg_lo_ss(Z_AXIS,ARG6)
            k7 = arg_lo_ss(Z_AXIS,ARG7)
            k8 = arg_lo_ss(Z_AXIS,ARG8)
            k9 = arg_lo_ss(Z_AXIS,ARG9)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            l2 = arg_lo_ss(T_AXIS,ARG2)
            l3 = arg_lo_ss(T_AXIS,ARG3)
            l4 = arg_lo_ss(T_AXIS,ARG4)
            l5 = arg_lo_ss(T_AXIS,ARG5)
            l6 = arg_lo_ss(T_AXIS,ARG6)
            l7 = arg_lo_ss(T_AXIS,ARG7)
            l8 = arg_lo_ss(T_AXIS,ARG8)
            l9 = arg_lo_ss(T_AXIS,ARG9)
            DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)


                  IF ( arg_1(i1,j1,k1,l1) .EQ. bad_flag(1) .OR. 
     .                 arg_2(i2,j2,k2,l2) .EQ. bad_flag(2) .OR. 
     .                 arg_3(i3,j3,k3,l3) .EQ. bad_flag(3) .OR. 
     .                 arg_4(i4,j4,k4,l4) .EQ. bad_flag(4) .OR. 
     .                 arg_5(i5,j5,k5,l5) .EQ. bad_flag(5) .OR. 
     .                 arg_6(i6,j6,k6,l6) .EQ. bad_flag(6) .OR.
     .                 arg_7(i7,j7,k7,l7) .EQ. bad_flag(7) .OR.
     .                 arg_8(i8,j8,k8,l8) .EQ. bad_flag(8) .OR.
     .                 arg_9(i9,j9,k9,l9) .EQ. bad_flag(9) ) THEN

                     result(i,j,k,l) = bad_flag_result

                  ELSE

                     result(i,j,k,l) = arg_1(i1,j1,k1,l1) + 
     .                  arg_2(i2,j2,k2,l2) +
     .                  arg_3(i3,j3,k3,l3) +
     .                  arg_4(i4,j4,k4,l4) +
     .                  arg_5(i5,j5,k5,l5) +
     .                  arg_6(i6,j6,k6,l6) +
     .                  arg_7(i7,j7,k7,l7) +
     .                  arg_8(i8,j8,k8,l8) +
     .                  arg_9(i9,j9,k9,l9)

                  END IF


                  l1 = l1 + arg_incr(T_AXIS,ARG1)
                  l2 = l2 + arg_incr(T_AXIS,ARG2)
                  l3 = l3 + arg_incr(T_AXIS,ARG3)
                  l4 = l4 + arg_incr(T_AXIS,ARG4)
                  l5 = l5 + arg_incr(T_AXIS,ARG5)
                  l6 = l6 + arg_incr(T_AXIS,ARG6)
                  l7 = l7 + arg_incr(T_AXIS,ARG7)
                  l8 = l8 + arg_incr(T_AXIS,ARG8)
                  l9 = l9 + arg_incr(T_AXIS,ARG9)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
               k2 = k2 + arg_incr(Z_AXIS,ARG2)
               k3 = k3 + arg_incr(Z_AXIS,ARG3)
               k4 = k4 + arg_incr(Z_AXIS,ARG4)
               k5 = k5 + arg_incr(Z_AXIS,ARG5)
               k6 = k6 + arg_incr(Z_AXIS,ARG6)
               k7 = k7 + arg_incr(Z_AXIS,ARG7)
               k8 = k8 + arg_incr(Z_AXIS,ARG8)
               k9 = l9 + arg_incr(Z_AXIS,ARG9)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
            j4 = j4 + arg_incr(Y_AXIS,ARG4)
            j5 = j5 + arg_incr(Y_AXIS,ARG5)
            j6 = j6 + arg_incr(Y_AXIS,ARG6)
            j7 = j7 + arg_incr(Y_AXIS,ARG7)
            j8 = j8 + arg_incr(Y_AXIS,ARG8)
            j9 = j9 + arg_incr(Y_AXIS,ARG9)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
         i3 = i3 + arg_incr(X_AXIS,ARG3)
         i4 = i4 + arg_incr(X_AXIS,ARG4)
         i5 = i5 + arg_incr(X_AXIS,ARG5)
         i6 = i6 + arg_incr(X_AXIS,ARG6)
         i7 = i7 + arg_incr(X_AXIS,ARG7)
         i8 = i8 + arg_incr(X_AXIS,ARG8)
         i9 = i9 + arg_incr(X_AXIS,ARG9)
 400  CONTINUE
      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END
