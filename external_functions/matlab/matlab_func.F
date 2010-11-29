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


      SUBROUTINE matlab_func_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,'test of MATLAB function calls')

      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'S')
      CALL ef_set_arg_unit(id, arg, 'PSU')
      CALL ef_set_arg_desc(id, arg, 'Salinity')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
      arg = 2
      CALL ef_set_arg_name(id, arg, 'T')
      CALL ef_set_arg_unit(id, arg, 'deg C')
      CALL ef_set_arg_desc(id, arg, 'Temperature')
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



      SUBROUTINE matlab_func_result_limits(id)

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

c      call ef_set_axis_limits(id, X_AXIS, 1, 1)


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



* In this subroutine we compute the result
*
      SUBROUTINE matlab_func_compute(id, arg_1, arg_2, result)

C ACM  xmem_subsc.cmn change to EF_mem_subsc.cmn'

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

C ACM  change m1lox to mem1lox, etc.

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .           mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL result(mem10lox:mem10hix, mem10loy:mem10hiy, 
     .           mem10loz:mem10hiz, mem10lot:mem10hit)
c
c Ahh, Ferret and FORTRAN. No dynamic memory allocation, 
c so set some array  limits for temporary storage and 
c hope for the best.
c
      INTEGER MAXY,MAXZ
      parameter(MAXY=1024,MAXZ=1024)
      REAL stemp(MAXY, MAXZ)
      REAL ttemp(MAXY, MAXZ)

C ACM  call ef_get_coordinates with REAL*8 array to get coord values

      REAL*8 ptemp(MAXZ)
      REAL ptemp4(MAXZ)
      REAL rtemp(MAXY, MAXZ)

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
      INTEGER i1, j1, j2, k2, k1, l1, ysize,  zsize
      LOGICAL firstTime

      firstTime = .true.

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      ysize = arg_hi_ss(Y_AXIS, ARG1) - arg_lo_ss(Y_AXIS, ARG1) + 1
      zsize = arg_hi_ss(Z_AXIS, ARG1) - arg_lo_ss(Z_AXIS, ARG1) + 1
c
c TODO -- Error checking
c 

c
c Get the depth axis for pressure
c

      CALL ef_get_coordinates(id, ARG1, Z_AXIS, 
     1       arg_lo_ss(Z_AXIS, ARG1), arg_hi_ss(Z_AXIS, ARG1), ptemp )


C ACM  Stick the coordinates in REAL arrays.  (also in the line above, a
C      Z_axis was typed as X_axis)

      DO l = 1, zsize
         ptemp4(l) = ptemp(l)
      ENDDO

      i1 = res_lo_ss(X_AXIS)
      DO i=arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1)
         l1 = res_lo_ss(T_AXIS)
         DO l=arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1)

c
c Copy slices to buffers
c 
            j2 = 1
            do j=arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
               k2 = 1
               do k=arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
                  stemp(j2,k2) = arg_1(i,j,k,l)
                  ttemp(j2,k2) = arg_2(i,j,k,l)
                  k2 = k2 + 1
               end do
               j2 = j2 + 1
            end do
            
c
c Get the results from MATLAB
c
            call matlab_func_demo(firstTime,stemp,ttemp,ptemp4,
     1            rtemp,ysize, zsize, bad_flag(1), MAXY, MAXZ)
            firstTime = .false.
c
c Copy the results from MATLAB into the Ferret buffer
c
            j2 = 1
            do j1=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
               k2 = 1
               do k1=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
c                  write(6,*) i1,j1,k1,l1,j2,k2,rtemp(j2, k2)
                  result(i1,j1,k1,l1) = rtemp(j2, k2)
                  k2 = k2 + 1
               end do
               j2 = j2 + 1
            end do
c
c Copy to results array
c
            
            l1 = l1 + 1
         end do
         i1 = i1 + 1
      end do
            
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
