* 
*  expndi_by.F
* 
*  Steve Hankin
*  1/16/2001
*  ACM v6.85 3/2013 Fix handling of bad-flags coming in on Arg 1
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
* 
*  Expand argument 1 by inserting a gap before each valid point in 
* argument 2 along axis I
*  Argument 2 may be multidimensional
 
*  In this subroutine we provide information about
*  the function.  The user configurable information 
*  consists of the following:
* 
*  descr              Text description of the function
* 
*  num_args           Required number of arguments
* 
*  axis_inheritance   Type of axis for the result
*                        ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                        CUSTOM          - user defined axis
*                        IMPLIED_BY_ARGS - same axis as the incoming argument
*                        NORMAL          - the result is normal to this axis
*                        ABSTRACT        - an axis which only has index values
* 
*  piecemeal_ok       For memory optimization:
*                        axes where calculation may be performed piecemeal
*                        ( YES, NO )
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


      SUBROUTINE expndi_by_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .       'Expand arg1 by inserting gaps before valid pts of arg 2' )

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT,        IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)


      arg = 1
      CALL ef_set_arg_name(id, arg, 'dat')
      CALL ef_set_arg_desc(id, arg, 'variable to gap-expand on I axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'mask')
      CALL ef_set_arg_desc(id, arg,
     .          'valid/bad points guiding gap insertions (multi-D)')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, YES, YES, YES, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'outsize')
      CALL ef_set_arg_desc(id, arg,
     .          'I size of result (constant)')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END






*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* lo_ss               lo subscript for an axis
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE expndi_by_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      REAL size

*
*     Use utility functions to get context information about the arguments.
*

      CALL ef_get_one_val(id, ARG3, size)

      CALL ef_set_axis_limits(id, X_AXIS, 1, INT(ABS(size)))

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



* 
*  In this subroutine we compute the result
* 
      SUBROUTINE expndi_by_compute(id, arg_1, arg_2, arg_3, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .           mem3lot:mem3hit, mem3loe:mem3hie, mem3lof:mem3hif)

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

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER imax

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      imax = res_hi_ss(X_AXIS)

* pre-fill the entire output result with missing value flags
*  (since each I-sequence in mask may have different number of valid pts)
      result = bad_flag_result

* fill in the points as specified by the mask
      n1 = arg_lo_ss(F_AXIS,ARG1)
      n2 = arg_lo_ss(F_AXIS,ARG2)
      DO 600 n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         m1 = arg_lo_ss(E_AXIS,ARG1)
         m2 = arg_lo_ss(E_AXIS,ARG2)
         DO 500 m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            l2 = arg_lo_ss(T_AXIS,ARG2)
            DO 400 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

               k1 = arg_lo_ss(Z_AXIS,ARG1)
               k2 = arg_lo_ss(Z_AXIS,ARG2)
               DO 300 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  j1 = arg_lo_ss(Y_AXIS,ARG1)
                  j2 = arg_lo_ss(Y_AXIS,ARG2)
                  DO 200 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

                     i1 = arg_lo_ss(X_AXIS,ARG1)
                     i  = res_lo_ss(X_AXIS)
                     DO 100 i2 = arg_lo_ss(X_AXIS,ARG2),
     .                           arg_hi_ss(X_AXIS,ARG2)

                        IF ( arg_2(i2,j2,k2,l2,m2,n2)
     .                       .NE. bad_flag(2) ) THEN
*                          Insert a gap
                           IF (i .GT. imax) CALL EF_BAIL_OUT(id,
     .                                          'Arg 3 size too small')
                           result(i,j,k,l,m,n) = bad_flag_result
                           i = i + 1
                        ENDIF
                        IF (i .GT. imax) CALL EF_BAIL_OUT(id,
     .                                          'Arg 3 size too small')
                        IF ( arg_1(i1,j1,k1,l1,m1,n1) .NE. bad_flag(1) ) 
     .                    result(i,j,k,l,m,n) = arg_1(i1,j1,k1,l1,m1,n1) 

                        i1 = i1 + arg_incr(X_AXIS,ARG1)
                        i = i + 1
 100                 CONTINUE

                     j1 = j1 + arg_incr(Y_AXIS,ARG1)
                     j2 = j2 + arg_incr(Y_AXIS,ARG2)
 200              CONTINUE

                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
                  k2 = k2 + arg_incr(Z_AXIS,ARG2)
 300           CONTINUE

               l1 = l1 + arg_incr(T_AXIS,ARG1)
               l2 = l2 + arg_incr(T_AXIS,ARG2)
 400        CONTINUE

            m1 = m1 + arg_incr(E_AXIS,ARG1)
            m2 = m2 + arg_incr(E_AXIS,ARG2)
 500     CONTINUE

         n1 = n1 + arg_incr(F_AXIS,ARG1)
         n2 = n2 + arg_incr(F_AXIS,ARG2)
 600  CONTINUE

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

