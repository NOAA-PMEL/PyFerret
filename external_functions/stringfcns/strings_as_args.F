*
* strings_as_args.F
*
* Ansley Manke
* 17-dec-2003
*
* This function takes two string arguments and one numeric argument.
* It returns a string result.  The result is the first N letters of the 
* first argument prepended to the string in the second argument, where N
* is the number of characters to take from the strings of arg1.
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


      SUBROUTINE strings_as_args_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .  'Demo Function: Prepend first NCHAR characters of string 1 onto string2')

      CALL ef_set_num_args(id, 3)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     . IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, STRING_RETURN)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'S1')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'string argument')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'S2')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'string argument')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'NCHAR')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, '# characters to take from S1')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type (id, arg, FLOAT_ARG)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE strings_as_args_compute(id, arg_1, arg_2, 
     .                                   arg_3, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result

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
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2
      INTEGER iarg, slen, slen1, slen2, nchar
      REAL value
      CHARACTER text1*10, text2*10, textout*20

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_one_val (id, 3, value)
      nchar = value

      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            k2 = arg_lo_ss(Z_AXIS,ARG2)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               l2 = arg_lo_ss(T_AXIS,ARG2)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
                  iarg = 1
 	          CALL EF_GET_STRING_ARG_ELEMENT (id, ARG1, arg_1, 
     .                                      i1,j1,k1,l1, slen1, text1)
                  CALL EF_GET_STRING_ARG_ELEMENT_LEN (id, iarg, arg_1, 
     .                                           i1,j1,k1,l1, slen1)
                  iarg = 2
 	          CALL EF_GET_STRING_ARG_ELEMENT (id, iarg, arg_2, 
     .                                      i2,j2,k2,l2, slen2, text2)
                  CALL EF_GET_STRING_ARG_ELEMENT_LEN (id, iarg, arg_2, 
     .                                           i1,j1,k1,l1, slen2)

                  iarg = 1
                  IF (slen1 .GE. nchar) THEN
                     textout = text1(1:nchar)//text2
                     slen = nchar + slen2
                  ELSE
                     textout = text1(1:slen1)//text2
                     slen = slen1 + slen2
                  ENDIF

                  CALL EF_PUT_STRING (textout, slen, result(i,j,k,l))

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
                  l2 = l2 + arg_incr(T_AXIS,ARG2)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
               k2 = k2 + arg_incr(Z_AXIS,ARG2)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
 400  CONTINUE
 

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
