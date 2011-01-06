*
* minmax.F
*
* Ansley Manke
* May 2009
*
* This function returns the min and max of a variable
* Faster than individually computing the min and max
* or running the STAT command.
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


      SUBROUTINE minmax_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Returns min and max of a var. result[i=1]=min,result[i=1]=max' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .     NORMAL, NORMAL, NORMAL)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'Variable')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
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
* loss               lo subscript for an axis
*
* hiss               hi subscript for an axis
*

      SUBROUTINE minmax_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V


      CALL ef_set_axis_limits(id, X_AXIS, 1, 2)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE minmax_compute(id, arg_1, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

	  INTEGER id

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
      REAL amin, amax

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      amin =    ABS(bad_flag(1))
      amax = -1*ABS(bad_flag(1))

      DO 400 l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
      DO 300 k1 = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
      DO 200 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
      DO 100 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)

         IF ( arg_1(i1,j1,k1,l1) .NE. bad_flag(1) ) THEN
            amin = MIN(amin, arg_1(i1,j1,k1,l1) )
            amax = MAX(amax, arg_1(i1,j1,k1,l1) )
         END IF

 100  CONTINUE
 200  CONTINUE
 300  CONTINUE
 400  CONTINUE

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)
      result(i,j,k,l) = amin
      result(i+1,j,k,l) = amax
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
