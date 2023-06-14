*
* length_of_strings.F
*
* Ansley Manke
* May 2002
*
* Returns length of strings in input array.
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


      SUBROUTINE length_of_strings_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V


      CALL ef_set_desc(id,
     . 'Demo Function:Returns the lengths of strings in A' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, FLOAT_RETURN)

      arg = 1
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
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
      SUBROUTINE length_of_strings_compute(id, arg_1, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l
      INTEGER i1, j1, k1, l1
      INTEGER atype, iarg, slen
      CHARACTER*100 errmsg

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_arg_type (id, 1, atype)
      IF (atype .NE. STRING_ARG) THEN
        errmsg = 'must call with string argument'
        GO TO 5000
      ENDIF

      iarg = 1

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)


         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                  CALL EF_GET_STRING_ARG_ELEMENT_LEN (id, ARG1, arg_1, 
     .                 i1,j1,k1,l1,  slen)

                  result(i,j,k,l) = slen

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE

      RETURN 
 5000 CALL EF_BAIL_OUT(id,errmsg)

      RETURN

      END
