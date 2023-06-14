*
* str_mask.F
*
* Ansley Manke 08/2013
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*
* This function takes a string variable and mask on the same axis.
* and returns the variable containing the strings where the mask is
* a valid value and a null string where the mask is bad-value
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


      SUBROUTINE str_mask_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 
     . 'Returns a variable expanded by the lengths given in arg 2' )
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, STRING_RETURN)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'STRING')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg,'string variable to mask')
      CALL ef_set_axis_influence_6d(id, arg, YES, YES, YES, YES, YES, YES)
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'MASK')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 
     .     'valid value where string is to be returned')
      CALL ef_set_axis_influence_6d(id, arg, YES, YES, YES, YES, YES, YES)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE str_mask_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'



      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit, 
     .           mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .           mem2loz:mem2hiz, mem2lot:mem2hit, 
     .           mem2loe:mem2hie, mem2lof:mem2hif)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshiz, memreslof:memreshif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(6), res_hi_ss(6), res_incr(6)
      INTEGER arg_lo_ss(6,1:EF_MAX_ARGS), arg_hi_ss(6,1:EF_MAX_ARGS),
     .     arg_incr(6,1:EF_MAX_ARGS)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l,m,n
      INTEGER slen
      CHARACTER*100 errtxt
      CHARACTER*2048 buff
      CHARACTER*1  null
      REAL str_mask

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)
      
      null = ''

* Put the requested features into the result

      DO n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
      DO m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
      DO l = res_lo_ss(Z_AXIS), res_hi_ss(T_AXIS)
      DO k = res_lo_ss(T_AXIS), res_hi_ss(Z_AXIS)
      DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
      DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

	 IF ( arg_2(i,j,k,l,m,n) .NE. bad_flag(2) ) THEN
            CALL ef_get_string_arg_element_6d(id, ARG1, arg_1, 
     .                      i,j,k,l,m,n, slen, buff)
            CALL EF_PUT_STRING(buff, slen, result(i,j,k,l,m,n))
	 ELSE
            CALL EF_PUT_STRING(null, 1, result(i,j,k,l,m,n))
         ENDIF  ! arg_2 good

      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
 9000 CALL EF_BAIL_OUT(id, errtxt)
      END
