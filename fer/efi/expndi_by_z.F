* 
*  expndi_by_z.F
* 
*  Steve Hankin
*  2/22/2001
*  ACM v6.85 3/2013 Fix handling of bad-flags coming in on Arg 1
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
* 
*  Expand argument 1 by createing a 2D XZ structure from it with the 
* separation between Z columns given by the location of non-missing values in
* argument 2. Thus the Z columns created are "ragged"
* Typical use is that arg 1 is a collection of vertical casts appended end-
* to-end and that arg 2 flags the start of each cast

*  Args 3 and 4 are constants -- the size of the X and Z axes, respectively
 
*  In this subroutine we provide information about
*  the function.  The user configurable information 
*  consists of the following:
* 
*  descr              Text description of the function
* 
*  num_args           Required number of arguments
* 
*  axis_inheritance   Type of axis for the result
*                        ( CUSTOM, IMPLIED_BY_Z_ARGS, NORMAL, ABSTRACT )
*                        CUSTOM          - user defined axis
*                        IMPLIED_BY_Z_ARGS - same axis as the incoming argument
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


      SUBROUTINE expndi_by_z_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .  'Expand arg1 on I into an XZ grid. (Typ. to split Z-profiles)' )

      CALL ef_set_num_args(id, 4)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT,        IMPLIED_BY_ARGS,
     .                                ABSTRACT,        IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'dat')
      CALL ef_set_arg_desc(id, arg, 'variable to expand on I axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, NO, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'mask')
      CALL ef_set_arg_desc(id, arg,
     .          'each valid point starts a new Z column')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, NO, YES, YES, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'n_profiles')
      CALL ef_set_arg_desc(id, arg,
     .          'I size of result (constant)')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'max_profile_len')
      CALL ef_set_arg_desc(id, arg,
     .          'K size of result (constant)')
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

      SUBROUTINE expndi_by_z_result_limits(id)

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

* output X axis size
      CALL ef_get_one_val(id, ARG3, size)
      CALL ef_set_axis_limits(id, X_AXIS, 1, INT(ABS(size)))

* output Z axis size
      CALL ef_get_one_val(id, ARG4, size)
      CALL ef_set_axis_limits(id, Z_AXIS, 1, INT(ABS(size)))

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



* 
*  In this subroutine we compute the result
* 
      SUBROUTINE expndi_by_z_compute(id, arg_1, arg_2, arg_3, arg_4,
     .                               result)

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
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .           mem4lot:mem4hit, mem4loe:mem4hie, mem4lof:mem4hif)

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
      INTEGER j1, k1, l1, m1, n1
      INTEGER j2, k2, l2, m2, n2
      INTEGER imax, kmax, iout, kout

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* make sure inputs are conformable
      IF ( (arg_lo_ss(X_AXIS,ARG1) .NE. arg_lo_ss(X_AXIS,ARG2))
     . .OR.(arg_hi_ss(X_AXIS,ARG1) .NE. arg_hi_ss(X_AXIS,ARG2)) ) CALL
     .           EF_BAIL_OUT(id, 'X limits of args 1 and 2 mismatched')
      IF ( (arg_lo_ss(Z_AXIS,ARG1) .NE. arg_hi_ss(Z_AXIS,ARG1))
     . .OR.(arg_lo_ss(Z_AXIS,ARG2) .NE. arg_hi_ss(Z_AXIS,ARG2)) ) CALL
     .           EF_BAIL_OUT(id, 'Z size of args 1 and 2 must be 1')

* limits on output arrays
      imax = res_hi_ss(X_AXIS)
      kmax = res_hi_ss(Z_AXIS)

* pre-fill the entire output result with missing value flags
*  (since each I-sequence in mask may have different number of valid pts)
      result = bad_flag_result

* fill in the points as specified by the mask
      k1 = arg_lo_ss(Z_AXIS,ARG1)
      k2 = arg_lo_ss(Z_AXIS,ARG2)

      n1 = arg_lo_ss(F_AXIS,ARG1)
      n2 = arg_lo_ss(F_AXIS,ARG2)
      DO 600 n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         m1 = arg_lo_ss(E_AXIS,ARG1)
         m2 = arg_lo_ss(E_AXIS,ARG2)
         DO 500 m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            l2 = arg_lo_ss(T_AXIS,ARG2)
            DO 400 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

               j1 = arg_lo_ss(Y_AXIS,ARG1)
               j2 = arg_lo_ss(Y_AXIS,ARG2)
               DO 200 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

                  iout = res_lo_ss(X_AXIS)
                  kout = res_lo_ss(Z_AXIS)
                  IF ( arg_2(arg_lo_ss(X_AXIS,ARG2),j2,k2,l2,m2,n2)
     .                 .NE. bad_flag(2) ) THEN
                     iout = iout - 1  ! will immediately increment
                  ELSE
                     kout = kout - 1  ! will immediately increment
                  ENDIF
                  DO 100 i = arg_lo_ss(X_AXIS,ARG1),
     .                       arg_hi_ss(X_AXIS,ARG1)
                     IF ( arg_2(i,j2,k2,l2,m2,n2)
     .                    .EQ. bad_flag(2) ) THEN
*    ... add another element to this Z column
                        IF (kout .GE. kmax) CALL EF_BAIL_OUT(id,
     .                                          'Arg 4 size too small')
                        kout = kout + 1
                     ELSE
*    ... start a new Z column
                        IF (iout .GE. imax) CALL EF_BAIL_OUT(id,
     .                                          'Arg 3 size too small')
                        iout = iout + 1
                        kout = res_lo_ss(Z_AXIS)
                     ENDIF
                     IF ( arg_1(i,j1,k1,l1,m1,n1) .NE. bad_flag(1) ) 
     .                 result(iout,j,kout,l,m,n)
     .                     = arg_1(i,j1,k1,l1,m1,n1)
 100              CONTINUE

                  j1 = j1 + arg_incr(Y_AXIS,ARG1)
                  j2 = j2 + arg_incr(Y_AXIS,ARG2)
 200           CONTINUE

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

