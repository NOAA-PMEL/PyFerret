*
* percent_good_t.F
*
* Jonathan Callahan
* Feb 19th 1998
*
* This function checks the input variable for bad data
* and returns 0 (no bad flags) or 1 (bad flags found)
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
* axis_reduction     Are the result axes 'reduced' with respect to the 
*                    argument axes from which they are inherited?
*                       ( RETAINED, REDUCED ) 
*                       RETAINED        - result axis has same extent as argument axis
*                       REDUCED         - result axis is reduced to a point
*                >>> The axis_reduction defaults to RETAINED and only needs
*                >>> to be applied when the result is reduced to a point but
*                >>> SET REGION information should still be applied to the
*                >>> external function arguments. (e.g. a function returning a status flag)
*                >>> In such a case the result axes should be IMPLIED_BY_ARGS
*                >>> and REDUCED.  (as opposed to NORMAL and RETAINED)
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


      SUBROUTINE percent_good_t_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . '(demonstration function) returns % good data at each time' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_axis_reduction(id, REDUCED, REDUCED, REDUCED,
     .     RETAINED)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_desc(id, arg, 'data to be checked')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE percent_good_t_compute(id, arg_1, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

	  INTEGER id, good, total

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*
* Note that we are looping on the argument subscripts rather than
* the result subscripts. (Looping on result subscripts is the normal 
* way).  That's because the result has axes which are normal and the
* loop count on those normal axes will be one.
*
      l = res_lo_ss(T_AXIS)
      DO 100 l1=arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
 
      good = 0
      total = 0

      i = res_lo_ss(X_AXIS)
      DO 400 i1=arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
 
         j = res_lo_ss(Y_AXIS)
         DO 300 j1=arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
 
            k = res_lo_ss(Z_AXIS)
            DO 200 k1=arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
 
                  total = total + 1

                  IF ( arg_1(i1,j1,k1,l1) .NE. bad_flag(1) ) THEN
 
                     good = good + 1
 
                  END IF
 
               k = k + res_incr(Z_AXIS)
 200        CONTINUE
 
            j = j + res_incr(Y_AXIS)
 300     CONTINUE
 
         i = i + res_incr(X_AXIS)
 400  CONTINUE
      
         result(i,j,k,l) = (good * 100.0) / total

         l = l + res_incr(T_AXIS)
 100  CONTINUE


 
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
