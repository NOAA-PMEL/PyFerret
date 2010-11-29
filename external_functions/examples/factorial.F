*
* factorial.F
*
* Kevin McHugh
* 13/05/2003
*
* Returns the product of the factorial.
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

      SUBROUTINE factorial_init(id)

c     INCLUDE '/home/ja12/data/mchugh/externfunc
c     .        /ef_utility/ferret_cmn/EF_Util.cmn'

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,'returns input factorial, 0 < input < 35 ' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

* All arguments should be on the same grid

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
**********************************************************************
      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE factorial_compute(id, arg_1, result)

c      INCLUDE 
c     .'/home/ja12/data/mchugh/externfunc/ef_utility/ferret_cmn/EF_Util.cmn'
c      INCLUDE 
c     .'/home/ja12/data/mchugh/externfunc/ef_utility/ferret_cmn/EF_mem_subsc.cmn'

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .     mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l,n, n_arg
      INTEGER i1, j1, k1, l1
      REAL factor

      factor=1

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)
      
      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)


                  IF ( arg_1(i1,j1,k1,l1) .EQ. bad_flag(1) ) THEN 

                     result(i,j,k,l) = bad_flag_result

                  ELSE IF ( arg_1(i1,j1,k1,l1) .GE. 35 ) THEN
                                          
                     result(i,j,k,l) = bad_flag_result

                  ELSE IF ( arg_1(i1,j1,k1,l1) .EQ. 0 ) THEN
                                          
                     result(i,j,k,l) = 1.

                  ELSE IF ( arg_1(i1,j1,k1,l1) .LT. 0 ) THEN
                                          
                     result(i,j,k,l) = bad_flag_result

                  ELSE
                     
                     factor = 1.
                     n_arg = arg_1(i1,j1,k1,l1)
                     DO 10 n = 2, n_arg
                        factor = factor * n
 10                     continue
                     result(i,j,k,l) = factor 

                  END IF


                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE
      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END
