*
*  extend.F
*
*  Simple test of axis extension.
*
*  Jonathan Callahan, October 1998
*

*
*  In this subroutine we provide information about the function
*
      SUBROUTINE extend_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,'text function for extending axes' )

      CALL ef_set_num_args(id, 1)
*
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS,        !^ Normaly 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)   !| set by
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)             !V default
*
*

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_desc(id, arg, 'anything')
      CALL ef_set_arg_unit(id, arg, 'un-normalized units')
      CALL ef_set_axis_extend(id, arg, X_AXIS, -1, 1)
      CALL ef_set_axis_extend(id, arg, Y_AXIS, -1, 1)
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

* 
*  In this subroutine we compute the result
* 
      SUBROUTINE extend_compute(id, a, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      REAL a(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .     mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .         arg_incr(4,EF_MAX_ARGS)
      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
*
      INTEGER i1, j1, k1, l1
      INTEGER i,  j,  k,  l

*
* get what we need
*
      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*
* =========
* Main Loop
* =========
* remember that the ARG1 axes are extended by -1,+1 in X and Y
*

             i1 = arg_lo_ss(X_AXIS,ARG1)+1
      DO 400 i  = res_lo_ss(X_AXIS),res_hi_ss(X_AXIS)

               j1 = arg_lo_ss(Y_AXIS,ARG1)+1
        DO 300 j  = res_lo_ss(Y_AXIS),res_hi_ss(Y_AXIS)

                k1 = arg_lo_ss(Z_AXIS,ARG1)
         DO 200 k  = res_lo_ss(Z_AXIS),res_hi_ss(Z_AXIS)

                 l1 = arg_lo_ss(T_AXIS,ARG1)
          DO 100 l  = res_lo_ss(T_AXIS),res_hi_ss(T_AXIS)


             IF ( a(i1  ,j1  ,k1,l1) .EQ. bad_flag(1).OR.
     .            a(i1+1,j1  ,k1,l1) .EQ. bad_flag(1)    ) THEN

                result(i,j,k,l) = bad_flag_result

             ELSE

                result(i,j,k,l) = a(i,j,k,l) * 0.5

             END IF


         l1 = l1 + arg_incr(T_AXIS,ARG1)
 100     CONTINUE

        k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200    CONTINUE

       j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300   CONTINUE

      i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE
      
         
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
