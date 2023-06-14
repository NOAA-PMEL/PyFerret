*
* maxstrlen.F
*
* Ansley Manke
* May 2002
*
* Returns max length of strings in input array.
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
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, maxstrlen )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )


      SUBROUTINE maxstrlen_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V


      CALL ef_set_desc(id,
     .  'Demo Function:Returns the max length of strings in A' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, NORMAL, NORMAL, NORMAL, NORMAL)
      CALL ef_set_result_type(id, FLOAT_RETURN)

      CALL ef_set_piecemeal_ok(id, YES, YES, YES, YES)

      arg = 1
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_desc(id, arg, 'Array of strings')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


* In this subroutine we compute the result
*
      SUBROUTINE maxstrlen_compute(id, arg_1, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)
      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER atype, iarg, slen
      CHARACTER*100 errmsg

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)

      CALL ef_get_arg_type (id, 1, atype)
      IF (atype .NE. STRING_ARG) THEN
        errmsg = 'must call with string argument'
        GO TO 5000
      ENDIF

      iarg = 1

      CALL EF_GET_STRING_ARG_MAX_LEN (id, iarg, arg_1, slen)
      result(res_lo_ss(X_AXIS), res_lo_ss(Y_AXIS), 
     .       res_lo_ss(Z_AXIS), res_lo_ss(T_AXIS))  = slen

      RETURN 
 5000 CALL EF_BAIL_OUT(id,errmsg)

      RETURN
      END
