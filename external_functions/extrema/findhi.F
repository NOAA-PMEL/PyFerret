*
* findhi.F
*
* Ansley Manke
* July 02, 2001
* jan 2002 -- change to use neighborhood in data units, not Ngridpoints.
*
* This function locates local extrema, returning "highs".
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


      SUBROUTINE findhi_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CHARACTER*120 fcn_descr

      WRITE (fcn_descr,100) 
      CALL ef_set_desc(id, fcn_descr)
  100 FORMAT('Find local maxima of var, return',
     .        ' HIGHS on Y axis. i=1 xpos,i=2 ypos,i=3 HI value')

      CALL ef_set_num_args(id, 3)
      CALL ef_set_axis_inheritance(id, ABSTRACT, ABSTRACT, 
     .			IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'var in X, Y and optionally Z, T')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'XRNG') 
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 
     .'Find local maxima in neighborhood +- XRNG, in data units')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'YRNG')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 
     .'Find local maxima in neighborhood +- YRNG, in data units')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* lo_ss               lo subscript for an axis
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE findhi_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo, my_hi, nx, ny

      INTEGER arg
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the argument.
*

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      CALL ef_set_axis_limits(id, X_AXIS, 1, 3)

      my_lo = 1
      my_hi = 1

      arg = 1
      nx = arg_hi_ss(X_AXIS,arg) - arg_lo_ss(X_AXIS,arg) + 1
      ny = arg_hi_ss(Y_AXIS,arg) - arg_lo_ss(Y_AXIS,arg) + 1
      my_hi = nx* ny	! an absolute upper bound

      CALL ef_set_axis_limits(id, Y_AXIS, my_lo, my_hi)
      my_lo = 1
      my_hi = 1

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
      SUBROUTINE findhi_work_size(id)

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
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER mx1, my1, mxy, mxdim

      INTEGER iwork
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mx1 = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      my1 = 1 + arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)

      mxdim = 3
      mxy = mx1 * my1

*  xyfield
      iwork = 1
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mx1, my1, 1, 1)

*  zmaxlist
      iwork = 2
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mxdim, mxy, 1, 1)

*  xax
      iwork = 3
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   2*mx1, 1, 1, 1)

*  yax
      iwork = 4
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   2*my1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE findhi_compute(id, arg_1, arg_2, arg_3, result, 
     .                          xyfield, zmaxlist, xax, yax)

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

      INTEGER i, j, k, l, i1,j1, k1, l1, arg, nx, ny, nzmax, ip, jp,
     .        neigh_x, neigh_y
      REAL    dx, dy, xsc, ysc

*  Dimension the work arrays.

      REAL xyfield(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL zmaxlist(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL*8 xax(wrk3lox:wrk3hix/2, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL*8 yax(wrk4lox:wrk4hix/2, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


*  Get the axis coordinates
      arg = 1
      CALL ef_get_coordinates(id, arg, X_AXIS, arg_lo_ss(X_AXIS,arg),
     .   arg_hi_ss(X_AXIS,arg), xax)
      CALL ef_get_coordinates(id, arg,Y_AXIS, arg_lo_ss(Y_AXIS,arg),
     .   arg_hi_ss(Y_AXIS,arg), yax)

      nx = arg_hi_ss(X_AXIS,arg) - arg_lo_ss(X_AXIS,arg) + 1
      ny = arg_hi_ss(Y_AXIS,arg) - arg_lo_ss(Y_AXIS,arg) + 1

      dx = xax(2,1,1,1) - xax(1,1,1,1)
      dy = yax(2,1,1,1) - yax(1,1,1,1)

*  Get the parameters neigh_x, neigh_y, to determine the region in which relative 
*  mins are found: 
C A RELATIVE MINIMUM (OR MAXIMUM) IS DEFINED TO BE THE LOWEST (OR HIGHEST) 
C POINT WITHIN A CERTAIN NEIGHBORHOOD OF THE POINT.  THE NEIGHBORHOOD USED 
C HERE IS + OR - neigh_x IN THE X DIRECTION AND + OR - neigh_y IN THE Y 
C DIRECTION

      CALL ef_get_one_val (id, ARG2, xsc)
      neigh_x = MAX(1, INT (xsc/dx))
      CALL ef_get_one_val (id, ARG3, ysc)
      neigh_y = MAX(1, INT (ysc/dy))

*  For each level and each time, call maxminmax to get the maxima.
*  Determine the coordinates of the max's and put with the max value
*  into the result

      k1 = arg_lo_ss(Z_AXIS,ARG1)
      DO 400 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

         l1 = arg_lo_ss(T_AXIS,ARG1)
         DO 300 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

            j1 = arg_lo_ss(Y_AXIS,ARG1)
	    DO 200 j = 1, ny
               i1 = arg_lo_ss(X_AXIS,ARG1)
               DO 100 i = 1, nx
                  xyfield(i,j,1,1) = arg_1(i1,j1,k1,l1)
                  i1 = i1 + arg_incr(X_AXIS,ARG1)
  100          CONTINUE
               j1 = j1 + arg_incr(Y_AXIS,ARG1)
  200       CONTINUE

            CALL MAXMINMAX (xyfield, nx, nx, ny, neigh_x, neigh_y, 
     .                      bad_flag(ARG1), zmaxlist, nzmax)

C  result(1,j,k,l) = X index of max
C  result(2,j,k,l) = Y index of max
C  result(3,j,k,l) = z value at max

            DO 210 j = 1, nzmax
               ip = INT(zmaxlist(1,j,1,1))
               jp = INT(zmaxlist(2,j,1,1))
               result(1,j,k,l) = xax(ip,1,1,1)
               result(2,j,k,l) = yax(jp,1,1,1)
               result(3,j,k,l) = zmaxlist(3,j,1,1)
 210        CONTINUE

C  Fill end of result with missing-data flags

            DO 230 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
               DO 220 j = nzmax+1, res_hi_ss(Y_AXIS)
                  result(i,j,k,l) = bad_flag_result
 220           CONTINUE
 230        CONTINUE

            l1 = l1 + arg_incr(T_AXIS,ARG1)
 300     CONTINUE

         k1 = k1 + arg_incr(Z_AXIS,ARG1)
 400  CONTINUE
      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
