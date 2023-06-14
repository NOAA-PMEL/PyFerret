*
* string_cat3.F
*
* Ansley Manke
* May 2006
*
*
* Returns concatenation of 3 strings
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
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, string_cat3 )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )


      SUBROUTINE string_cat3_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V


      CALL ef_set_desc(id,
     .  'Demo Function: Returns concatenation of 3 strings' )

      CALL ef_set_num_args(id, 3)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, STRING_RETURN)

      arg = 1
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_desc(id, arg, 'string1')
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      arg = 2
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_desc(id, arg, 'string2')
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      arg = 3
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
      CALL ef_set_arg_name(id, arg, 'C')
      CALL ef_set_arg_desc(id, arg, 'string3')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


* In this subroutine we compute the result
*
      SUBROUTINE string_cat3_compute(id, arg_1, arg_2, arg_3, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id


      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .           mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .           mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit)

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
      INTEGER i1, j1, k1, l1, i2, j2, k2, l2, i3, j3, k3, l3
      INTEGER atype, iarg, slen, slen1, slen2, slen3
      CHARACTER*100 errmsg, str1*255, str2*255, str3*255

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_arg_type (id, ARG1, atype)
      IF (atype .NE. STRING_ARG) THEN
        errmsg = 'must call with string arguments'
        GO TO 5000
      ENDIF

      CALL ef_get_arg_type (id, ARG2, atype)
      IF (atype .NE. STRING_ARG) THEN
        errmsg = 'must call with string arguments'
        GO TO 5000
      ENDIF

      CALL ef_get_arg_type (id, ARG3, atype)
      IF (atype .NE. STRING_ARG) THEN
        errmsg = 'must call with string arguments'
        GO TO 5000
      ENDIF

      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      i3 = arg_lo_ss(X_AXIS,ARG3)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         j3 = arg_lo_ss(Y_AXIS,ARG3)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            k2 = arg_lo_ss(Z_AXIS,ARG2)
            k3 = arg_lo_ss(Z_AXIS,ARG3)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               l2 = arg_lo_ss(T_AXIS,ARG2)
               l3 = arg_lo_ss(T_AXIS,ARG3)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
                     
                  CALL EF_GET_STRING_ARG_ELEMENT (id, ARG1, arg_1, 
     .                                      i1,j1,k1,l1, slen1, str1)
                  CALL EF_GET_STRING_ARG_ELEMENT (id, ARG2, arg_2, 
     .                                      i2,j2,k2,l2, slen2, str2)
                  CALL EF_GET_STRING_ARG_ELEMENT (id, ARG3, arg_3, 
     .                                      i3,j3,k3,l3, slen3, str3)
                  
                     slen = slen1 + slen2 + slen3
                     CALL EF_PUT_STRING (
     .                  str1(:slen1)//str2(:slen2)//str3(:slen3), 
     .                  slen, result(i,j,k,l))

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
                  l2 = l2 + arg_incr(T_AXIS,ARG2)
                  l3 = l3 + arg_incr(T_AXIS,ARG3)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
               k2 = k2 + arg_incr(Z_AXIS,ARG2)
               k3 = k3 + arg_incr(Z_AXIS,ARG3)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
         i3 = i3 + arg_incr(X_AXIS,ARG3)
 400  CONTINUE
      
      RETURN 
 5000 CALL EF_BAIL_OUT(id,errmsg)

      RETURN

      END
