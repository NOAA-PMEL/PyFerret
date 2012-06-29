*
* avet.F
*
* Jonathan Callahan
* Feb 19th 1998
*
* Returns the argument averaged over the T axis
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


      SUBROUTINE avet_init(id)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CALL ef_set_desc(id,
     .        '(demonstration function) returns the time average')

      CALL ef_set_num_args(id, 1)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, NORMAL,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, YES, YES, YES, NO, YES, YES)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_desc(id, arg,
     .        'data to be averaged over the time axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, NO, YES, YES)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE avet_compute(id, arg_1, result)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)

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

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER i,  j,  k,  l,  m,  n
      INTEGER i1, j1, k1, l1, m1, n1
      REAL    datasum
      REAL    datacount

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

       m1 = arg_lo_ss(E_AXIS,ARG1)
       DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

        k1 = arg_lo_ss(Z_AXIS,ARG1)
        DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

          i1 = arg_lo_ss(X_AXIS,ARG1)
          DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

*
* Loop through time (using the argument indices) and calculate a running sum.
*
           datasum = 0.0
           datacount = 0.0
           DO 400 l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
              IF ( arg_1(i1,j1,k1,l1,m1,n1) .NE. bad_flag(ARG1) ) THEN
                 datasum = datasum + arg_1(i1,j1,k1,l1,m1,n1)
                 datacount = datacount + 1.0
              ENDIF
 400       CONTINUE

*          Result is normal to the time axis, thus only one index
*          on the result's time dimension
           l = res_lo_ss(T_AXIS)

*
* If we had any good data, calculate the average;
* otherwise set the result to undefined.
*
           IF ( datacount .GT. 0.0 ) THEN
              result(i,j,k,l,m,n) = datasum / datacount
           ELSE
              result(i,j,k,l,m,n) = bad_flag_result
           ENDIF

           i1 = i1 + arg_incr(X_AXIS,ARG1)
 100      CONTINUE

          j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200     CONTINUE

         k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300    CONTINUE

        m1 = m1 + arg_incr(E_AXIS,ARG1)
 500   CONTINUE

       n1 = n1 + arg_incr(F_AXIS,ARG1)
 600  CONTINUE

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN
      END
