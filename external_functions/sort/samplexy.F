*
* samplexy.F
*
* Ansley Manke
* APR 1999
*
* This function samples 4-d data on the x and y axes indicated by args 2 and 3
* 4/5/99  Ansley Manke 
*         Result is abstract on the x axis, normal on the y axis,
*         and keeps the z and t axes of the input 4-d data.
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


      SUBROUTINE samplexy_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled at a collection of x, y points')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .     NORMAL, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 2)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'variable (x,y,z,t) to sample')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 'X indices of sample points')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 'Y indices of sample points')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* loss               lo subscript for an axis
*
* hiss               hi subscript for an axis
*

      SUBROUTINE samplexy_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo_l, my_hi_l

*     Use utility functions to get context information about the 
*     1st argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG2) - arg_lo_ss(X_AXIS, ARG2) + 1
      ny = arg_hi_ss(Y_AXIS, ARG2) - arg_lo_ss(Y_AXIS, ARG2) + 1
      nz = arg_hi_ss(Z_AXIS, ARG2) - arg_lo_ss(Z_AXIS, ARG2) + 1
      nt = arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) + 1

      my_lo_l = 1
      my_hi_l = max(nx,ny,nz,nt)

      CALL ef_set_axis_limits(id, X_AXIS, my_lo_l, my_hi_l)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE samplexy_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_lens(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER mxdat, mydat, mx2, my2

      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mxdat = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      mydat = 1 + arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)

      mx2 = mxdat* 2
      my2 = mydat* 2

*  xaxdat
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, mx2, 1, 1, 1)

*  yaxdat
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, my2, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE samplexy_compute(id, arg_1, arg_2, arg_3, result, 
     .       xaxdat, yaxdat)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .     mem3lot:mem3hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .      memresloz:memreshiz, memreslot:memreshit)


* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      COMMON /STOR/ mxdat, mydat
      INTEGER mxdat, mydat

*  Set up work arrays

      REAL*8 xaxdat(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 yaxdat(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)

      INTEGER imatch
      integer i1,j1,k1,l1
      integer i2,j2,k2,l2
      integer i3,j3,k3,l3

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Get x and y coordinates of the data array.

      CALL ef_get_coordinates(id, ARG1, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), xaxdat)
      CALL ef_get_coordinates(id, ARG1, Y_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), yaxdat)

      i2 = arg_lo_ss(X_AXIS,ARG2)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      k2 = arg_lo_ss(Z_AXIS,ARG2)
      l2 = arg_lo_ss(Y_AXIS,ARG2)

      i3 = arg_lo_ss(X_AXIS,ARG3)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)

*  For each (xpt,ypt) pair, search the data array arg_1 and match its (x,y)
*  coordinates.  Keep the result in result(i_point,_,k,l) for all k,l in
*  the data array.

      j = res_lo_ss(Y_AXIS)
      DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         imatch = 0

         DO 200 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1) 
         DO 220 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1) 

            IF (arg_2(i2,j2,k2,l2) .eq. xaxdat(i1,1,1,1)  .AND.
     .          arg_3(i3,j3,k3,l3) .eq. yaxdat(j1,1,1,1) ) THEN

               imatch = 1

               k1 = arg_lo_ss(Z_AXIS,ARG1)
               DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  l1 = arg_lo_ss(T_AXIS,ARG1)
                  DO 300 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                     result(i,j,k,l) = arg_1(i1,j1,k1,l1)

                     l1 = l1 + arg_incr(T_AXIS,ARG1)
 300              CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 400           CONTINUE

            ENDIF


 220     CONTINUE
 200     CONTINUE

*  Data coordinates do not match (xpts,ypts) coordinates

         IF (imatch .eq. 0) then
            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 450 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 350 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                  result(i,j,k,l) = bad_flag_result

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 350           CONTINUE

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
 450        CONTINUE
         ENDIF


         i2 = i2 + arg_incr(X_AXIS,ARG2)
         j2 = j2 + arg_incr(Y_AXIS,ARG2)
         k2 = k2 + arg_incr(Z_AXIS,ARG2)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         j3 = j3 + arg_incr(Y_AXIS,ARG3)
 100  CONTINUE

      RETURN
      END
