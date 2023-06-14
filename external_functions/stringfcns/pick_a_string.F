*
* pick_a_string.F
*
* Ansley Manke
* 15-dec-2003
*
* This function chooses an element element of the first
* argument, which is a 1-D string array, and we choose
* the element given by the (scalar) second argument
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


      SUBROUTINE pick_a_string_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Demo Function: sets result equal to one element of input' )

      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, STRING_RETURN)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, '1-D string argument')
      CALL ef_set_axis_influence(id, arg, NO, YES, YES, YES)
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      
      arg = 2
      CALL ef_set_arg_name(id, arg, 'N')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id,arg,'Pick Nth element in the X direction')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type (id, arg, FLOAT_ARG)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


      SUBROUTINE pick_a_string_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*     You could use utility functions at this point to get 
*     context information about the other arguments.  This info 
*     could be used to update the custom or abstract axis lo and hi
*     indices.

      call ef_set_axis_limits(id, X_AXIS, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END




*
* In this subroutine we compute the result
*
      SUBROUTINE pick_a_string_compute(id, arg_1, arg_2, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem1hix, mem2loy:mem1hiy, 
     .           mem2loz:mem1hiz, mem2lot:mem1hit)
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
      INTEGER indx, nx
      REAL value
      CHARACTER*40 err_msg
      
      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_one_val(id, 2, value)
      indx = value
      nx = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1
      IF (indx .LT. 1 .OR. indx .GT. nx) THEN
         WRITE (err_msg, *) 'Argument 2 too small or large. NX=', nx
         CALL EF_BAIL_OUT (id, err_msg)
      ENDIF

      i1 = indx

      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

c                  CALL EF_GET_STRING_ARG_ELEMENT (id, ARG1, arg_1, 
c     .                               i1,j1,k1,l1, slen, textstring)
c                  CALL EF_PUT_STRING (textstring, slen, result(i,j,k,l))

                   CALL EF_PUT_STRING_PTR(arg_1(i1,j1,k1,l1),
     .                            result(i,j,k,l) )

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

 400   CONTINUE

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
