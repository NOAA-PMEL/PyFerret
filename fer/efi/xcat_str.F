*
* xcat_str.F
*
* Ansley Manke
* Nov 2008
*
* This function returns the X values of the two variables,
* concatenated into one long list on an abstract x axis. Other
* axes are unchanged.
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


      SUBROUTINE xcat_str_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*100 descrip

      WRITE (descrip, 100)
  100 FORMAT ('Concatenates the X values of two string variables ',
     .        'into one list on an abstract X axis')
      CALL ef_set_desc(id, descrip)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT,        IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_result_type(id, STRING_RETURN)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'First variable to concatenate')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'Second variable to concatenate')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, YES, YES, YES, YES)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


      SUBROUTINE xcat_str_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER nx

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = (arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1) +
     .     (arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1)

      CALL ef_set_axis_limits(id, X_AXIS, 1, nx)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE xcat_str_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy,
     .           mem1loz:mem1hiz, mem1lot:mem1hit,
     .           mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .           mem2loz:mem2hiz, mem2lot:mem2hit,
     .           mem2loe:mem2hie, mem2lof:mem2hif)

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

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l, m, n, slen
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      CHARACTER*512 buff

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      i = res_lo_ss(X_AXIS)
      DO 100 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)

         n = res_lo_ss(F_AXIS)
         DO 600 n1 = arg_lo_ss(F_AXIS,ARG1), arg_hi_ss(F_AXIS,ARG1)

         m = res_lo_ss(E_AXIS)
         DO 500 m1 = arg_lo_ss(E_AXIS,ARG1), arg_hi_ss(E_AXIS,ARG1)

         l = res_lo_ss(T_AXIS)
         DO 400 l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)

         k = res_lo_ss(Z_AXIS)
         DO 300 k1 = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)

         j = res_lo_ss(Y_AXIS)
         DO 200 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)

            CALL EF_GET_STRING_ARG_ELEMENT_6D(id, ARG1, arg_1,
     .                             i1, j1, k1, l1, m1, n1, slen, buff)
            CALL EF_PUT_STRING(buff, slen, result(i,j,k,l,m,n))

            j = j + res_incr(Y_AXIS)
 200     CONTINUE

            k = k + res_incr(Z_AXIS)
 300     CONTINUE

            l = l + res_incr(T_AXIS)
 400     CONTINUE

            m = m + res_incr(E_AXIS)
 500     CONTINUE

            n = n + res_incr(F_AXIS)
 600     CONTINUE

         i = i + res_incr(X_AXIS)
 100  CONTINUE

* concatenate in X - dont restart count of index I

      DO 110 i2 = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)

         n = res_lo_ss(F_AXIS)
         DO 610 n2 = arg_lo_ss(F_AXIS,ARG2), arg_hi_ss(F_AXIS,ARG2)

         m = res_lo_ss(E_AXIS)
         DO 510 m2 = arg_lo_ss(E_AXIS,ARG2), arg_hi_ss(E_AXIS,ARG2)

         l = res_lo_ss(T_AXIS)
         DO 410 l2 = arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2)

         k = res_lo_ss(Z_AXIS)
         DO 310 k2 = arg_lo_ss(Z_AXIS,ARG2), arg_hi_ss(Z_AXIS,ARG2)

         j = res_lo_ss(Y_AXIS)
         DO 210 j2 = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)

            CALL EF_GET_STRING_ARG_ELEMENT_6D(id, ARG2, arg_2,
     .                             i2, j2, k2, l2, m2, n2, slen, buff)
            CALL EF_PUT_STRING(buff, slen, result(i,j,k,l,m,n))

            j = j + res_incr(Y_AXIS)
 210     CONTINUE

            k = k + res_incr(Z_AXIS)
 310     CONTINUE

            l = l + res_incr(T_AXIS)
 410     CONTINUE

            m = m + res_incr(E_AXIS)
 510     CONTINUE

            n = n + res_incr(F_AXIS)
 610     CONTINUE

         i = i + res_incr(X_AXIS)
 110  CONTINUE

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END
