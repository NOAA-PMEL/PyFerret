*
*  scat2gridgauss_xz.F
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* Ansley Manke
* July 28 1998
*  revised 2/00 to use work arrays
* June 2000 ACM pass 4 gridding arguments:  x and z radius of influence,
*               and x and z cutoffs. 
*
*
* Returns variable interpolated onto an equally-spaced X-Y grid.  
* Input is scattered triples: (x, y, f(x,y)); may be functions of z and/or time.
* Output is gridded data in x, y, z, and time.  Calls routine "gausswt2".
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


      SUBROUTINE  scat2gridgauss_xz_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Use Gaussian weighting to grid scattered data to an ',
     .        'XZ grid.')
      CALL ef_set_desc(id, fcn_desc)


      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .      NORMAL, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

* XZ grid is determined by arguments 4 and 5, the result's x and z axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'X coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'ZPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Z coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      CALL ef_set_arg_desc(id, arg, 
     .     'F(X,Z)   3rd component of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAX')
      CALL ef_set_arg_desc(id, arg, 'X axis of output grid')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'ZAX')
      CALL ef_set_arg_desc(id, arg, 'Z axis of output grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'XSCALE')
      WRITE (fcn_desc, 30)
      CALL ef_set_arg_desc(id, arg, fcn_desc)
   30 FORMAT ('Radius of influence in X',
     .        ' direction, data units (e.g. km or lon)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'ZSCALE')
      WRITE (fcn_desc, 40)
      CALL ef_set_arg_desc(id, arg, fcn_desc)
   40 FORMAT ('Radius of influence in Z',
     .        ' direction, data units (e.g. m or km)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)


      arg = 8
      CALL ef_set_arg_name(id, arg, 'XCUTOFF')
      WRITE (fcn_desc, 50)
      CALL ef_set_arg_desc(id, arg, fcn_desc)
   50 FORMAT ('CUTOFF for wt fcn in X direction: cutoff=2 limits the ',
     .        'search to min wt=e**-4')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 9
      CALL ef_set_arg_name(id, arg, 'ZCUTOFF')
      WRITE (fcn_desc, 60)
      CALL ef_set_arg_desc(id, arg, fcn_desc)
   60 FORMAT ('CUTOFF for wt fcn in Z direction: cutoff=2 limits the',
     .        'search to min wt=e**-4')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END
*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE scat2gridgauss_xz_work_size(id)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER nxout, nzout, nx2, nz2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4)
      nzout = 1 + arg_hi_ss(Z_AXIS,ARG5) - arg_lo_ss(Z_AXIS,ARG5)

      nx2 = nxout* 2
      nz2 = nzout* 2

* xax  output x axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nx2, 1, 1, 1)

* zax  output z axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, nz2, 1, 1, 1)

* grid  work array - gridded data. 
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, 
     .                             nxout, nzout, 1, 1)

* wate - weights.
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, 
     .                             nxout, nzout, 1, 1)

************************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*

      SUBROUTINE scat2gridgauss_xz_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, result, 
     .     xax, zax, grid, wate)



*  arg_1  xpts \
*  arg_2  zpts  > Scattered x,z,f(x,z) triples to be gridded.
*  arg_3  fpts /    Can be fcn of z,t
*  arg_4  X axis of new grid
*  arg_5  Z axis of new grid
*  arg_6  interpolation parameter   xscale
*  arg_7  interpolation parameter   zscale
*  arg_8  interpolation parameter   xcutoff
*  arg_9  interpolation parameter   zcutoff

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id
      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .     mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .     mem4lot:mem4hit)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz, 
     .     mem5lot:mem5hit)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, mem6loz:mem6hiz, 
     .     mem6lot:mem6hit)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, mem7loz:mem7hiz,
     .      mem7lot:mem7hit)
      REAL arg_8(mem8lox:mem8hix, mem8loy:mem8hiy, mem8loz:mem8hiz,
     .      mem8lot:mem8hit)
      REAL arg_9(mem9lox:mem9hix, mem9loy:mem9hiy, mem9loz:mem9hiz,
     .      mem9lot:mem9hit)

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
      INTEGER i, j, k, l, m, n, nm
      INTEGER i1, i2, i1n, i2n
      INTEGER nxpts, nzpts, nscat
      INTEGER i4, i4n, j5, j5n
      INTEGER nxsize, nzsize
      REAL xsc, zsc, tsc 
      REAL xcutoff, zcutoff
      REAL tt, t1, tf
      INTEGER iwflag, ii3, jj3, kk3, ll3

      INTEGER nx, nz, nxaxis, nzaxis
      REAL x1, z1, xf, zf
      REAL xx, zz, val

*  Dimension the work arrays

      REAL*8 xax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 zax(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL grid(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL wate(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)

      CHARACTER*100 errtxt

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, or Z axis of the input arguments.

      DO 100 m = X_AXIS, Z_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG1)
            i1n = arg_hi_ss(m,ARG1)
            if (i1n-i1 .NE. 0) nxpts = 1 + (i1n - i1)
         ENDIF
 100  CONTINUE

      DO 110 m = X_AXIS, Z_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG2)
            i2n = arg_hi_ss(m,ARG2)
            if (i2n-i2 .NE. 0) nzpts = 1 + (i2n - i2)
         ENDIF
 110  CONTINUE

      IF (nzpts .NE. nxpts ) GOTO 999

      nscat = nxpts

*  Compute number of points in output axes.

      i4 = arg_lo_ss(X_AXIS,ARG4)
      i4n = arg_hi_ss(X_AXIS,ARG4)

      j5 = arg_lo_ss(Z_AXIS,ARG5)
      j5n = arg_hi_ss(Z_AXIS,ARG5)

      nx = 1 + (i4n - i4)
      nz = 1 + (j5n - j5)
      nxsize = nx
      nzsize = nz

      call ef_get_coordinates(id, ARG4, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG4), arg_hi_ss(X_AXIS, ARG4), xax)

      call ef_get_coordinates(id, ARG5, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, ARG5), arg_hi_ss(Z_AXIS, ARG5), zax)

*  Set start, end, and delta for output axes. 
*  	x1,z1=west/top edge of grid (center of 1st box in data units)
*  	xf,zf=east/bottom edge of grid (center of final box)


      x1 = xax(1,1,1,1)
      z1 = zax(1,1,1,1)

      xf = xax(nx,1,1,1)
      zf = zax(nz,1,1,1)

*  Get interpolation parameters: mapping scales (data units)

      xsc  = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6), 
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6))

      zsc  = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7), 
     .            arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7))

*  And cutoff parameters:

      xcutoff  = arg_8(arg_lo_ss(X_AXIS,ARG8), arg_lo_ss(Y_AXIS,ARG8), 
     .            arg_lo_ss(Z_AXIS,ARG8), arg_lo_ss(T_AXIS,ARG8))

      zcutoff  = arg_9(arg_lo_ss(X_AXIS,ARG9), arg_lo_ss(Y_AXIS,ARG9), 
     .            arg_lo_ss(Z_AXIS,ARG9), arg_lo_ss(T_AXIS,ARG9))

*  Time parameters for subroutine gausswt.  Calling with 1 time.

      nm = 1
      tt = 1.
      t1 = 1.
      tf = 1.
      tsc = 1. 

*  iwflag=1 for time wrapping; 0 for no wrapping

      iwflag = 0

      nxaxis = arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4) + 1
      nzaxis = arg_hi_ss(Z_AXIS,ARG5) - arg_lo_ss(Z_AXIS,ARG5) + 1

*  Initialize sums of values and weights.

      DO j = 1, nzaxis
         DO i = 1, nxaxis
            grid(i,j,1,1) = 0.
         ENDDO
      ENDDO

      DO j = 1, nzaxis
         DO i = 1, nxaxis
            wate(i,j,1,1) = 0.
         ENDDO
      ENDDO

*  Compute result at each time, and each depth.


      DO 510 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS), 
     .                       max(1,res_incr(T_AXIS))
         DO 500 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS), 
     .                       max(1,res_incr(Y_AXIS))

*  Loop over x and z, compute the weighted sums for gaussian-weighted mapping
*  onto the grid.  Lat and longitude may be on the X,Y or Z axis of ARG1 and
*  ARG2: send them to a subroutine which collapses the extra dimensions so the
*  data value can be found.


         ii3 = arg_lo_ss(X_AXIS,ARG3)
         jj3 = arg_lo_ss(Y_AXIS,ARG3)
         kk3 = arg_lo_ss(Z_AXIS,ARG3)
         ll3 = arg_lo_ss(T_AXIS,ARG3) - 1 + l
         if (arg_lo_ss(T_AXIS,ARG3) .eq. ef_unspecified_int4) 
     .       ll3 = ef_unspecified_int4
         DO 300 n = 1, nscat

            CALL pickoutxyz (arg_1, arg_2, arg_3(ii3,jj3,kk3,ll3), 
     .                    n, xx, zz, val)

            IF ( val  .NE. bad_flag(ARG3) )
     .      CALL gausswt2 (xx, zz, tt, val, grid, wate, nx, nz, nm, 
     .                    x1, z1, t1, xf, zf, tf, xsc, zsc, tsc, 
     .                    xcutoff, zcutoff, iwflag, nxaxis, nzaxis) 

 300     CONTINUE

*  Put gridded z into result variable, dividing by summed weights. (as in
*  gaussfin, but indices needn't start at 1)


         DO 410 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS), 
     .                       max(1,res_incr(X_AXIS))
         DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS), 
     .                       max(1,res_incr(Z_AXIS))

            result(i,j,k,l) = 0.
            IF ( wate(i,k,1,1) .gt. 0.) THEN
               result(i,j,k,l) = grid(i,k,1,1)/ wate(i,k,1,1)
            ELSE
               result(i,j,k,l) = bad_flag_result
            ENDIF

 400        CONTINUE
 410     CONTINUE

 500  CONTINUE
 510  CONTINUE

      RETURN

999   CONTINUE

   20 FORMAT('Input x, z have different # of points', 2i8)

      WRITE (errtxt,20) nxpts, nzpts

      CALL EF_BAIL_OUT(id, errtxt)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END
