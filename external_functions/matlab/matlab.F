*
* matlab.F
*
* Joe Sirott
*
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


      SUBROUTINE matlab_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,'test of MATLAB (3D Surface visualization)')

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .     NORMAL, NORMAL, NORMAL)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'X')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'Variable to plot')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



      SUBROUTINE matlab_result_limits(id)

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



* In this subroutine we compute the result
*
      SUBROUTINE matlab_compute(id, arg_1, result)

C ACM  xmem_subsc.cmn change to EF_mem_subsc.cmn'

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'
      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

C ACM  change m1lox to mem1lox, etc.

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .     mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(mem10lox:mem10hix, mem10loy:mem10hiy, 
     .     mem10loz:mem10hiz, mem10lot:mem10hit)

* TODO -- what if coord length is greater than these values??

C ACM  call ef_get_coordinates with REAL*8 arrays to get coord values

      REAL*8 xcoords(8192), ycoords(8192)

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
      INTEGER i1, j1, k1, l1, xsize,  ysize

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_coordinates(id, ARG1, X_AXIS, 
     1       arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), xcoords ) 

      CALL ef_get_coordinates(id, ARG1, Y_AXIS, arg_lo_ss(Y_AXIS, ARG1),
     1        arg_hi_ss(Y_AXIS, ARG1), ycoords ) 

      xsize = arg_hi_ss(X_AXIS, ARG1) - arg_lo_ss(X_AXIS, ARG1) + 1
      ysize = arg_hi_ss(Y_AXIS, ARG1) - arg_lo_ss(Y_AXIS, ARG1) + 1

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
                  result(i,j,k,l) = 0
                  l1 = l1 + arg_incr(T_AXIS,ARG1)
               END DO
               k1 = k1 + arg_incr(Z_AXIS,ARG1)
            END DO
            j1 = j1 + arg_incr(Y_AXIS,ARG1)
         END DO
         i1 = i1 + arg_incr(X_AXIS,ARG1)
      END DO
      call matlab_demo(arg_1, xcoords, xsize, ycoords, ysize, 
     1      bad_flag(1))
     
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
