*
*  scat2gridgauss_xy.F
*
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
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND Anx EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR Anx SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
* Ansley Manke
* July 28 1998
*  revised 2/00 to use work arrays
* June 2000 ACM pass 4 gridding arguments:  y and z radius of influence,
*               and y and z cutoffs. 
*
* Nov 13, 2000 1) Allow modulo axes: if modulo take points from other end 
*                 to use in gridding each end
*              2) Check that the scattered points are listed on the I,J,or
*                 K axis only, as they may be functions of time.
*  (12/1/2000) 3) If the destination axis is modulo, treat the scattered
*                 points as modulo too.
*
*   12/7/2000  Add error checking on gridding parameters
*              X and Y scattered points are 1-D lists on any axis.
*
*    5/2001    Let F(x,y) be a function of Z and/or T
* v5.41 5/02   Fix bug (typo); Gridding to modulo X axis wasnt done.

*  11/12/2002  Remove option 2) above: the spatial location of the scattered points
*              is a 1-D list.
*   2/ 9/2005  When we fixed the argument limit machinery this fcn
*              breaks if the output axes are limited by some context.
*              use ef_set_axis_extend to force the regridding to be
*              done on the unrestricted output grid.


* Returns variable interpolated onto an equally-spaced X-Y grid.  
* Input is scattered triples: (x, y, f(x,y)); f may be function of time, z 
* Output is gridded data in x, y and time.  Calls routine "gausswt".
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


      SUBROUTINE scat2gridgauss_xy_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Use Gaussian weighting to grid scattered data to an ',
     .        'XY grid.')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .      IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

* Horizontal grid is determined by arguments 4 and 5, the result's x and y axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'X coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Y coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      WRITE (buff,20) 
   20 FORMAT ('F(X,Y) 3rd component of scattered input triples. ',
     .  'May be fcn of Z and/or T')
      CALL ef_set_arg_desc(id, arg, buff)
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'X axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)
      CALL ef_set_axis_extend (id, arg, X_AXIS, ef_unspecified_int4,
     .                         ef_unspecified_int4)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'YAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'Y axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, YES, NO, NO)
      CALL ef_set_axis_extend (id, arg, Y_AXIS, ef_unspecified_int4, 
     .                         ef_unspecified_int4)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'XSCALE')
      WRITE (buff, 30)
      CALL ef_set_arg_desc(id, arg, buff)
   30 FORMAT 
     . ('mapping scale in X direction, data units (e.g. km or lon)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'YSCALE')
      WRITE (buff, 40)
      CALL ef_set_arg_desc(id, arg, buff)
   40 FORMAT 
     .  ('mapping scale in Y direction, data units (e.g. km or lat)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 8
      CALL ef_set_arg_name(id, arg, 'CUTOFF')
      WRITE (buff, 50)
      CALL ef_set_arg_desc(id, arg, buff)
   50 FORMAT 
     .  ('Weight cutoff: Use pts within cutoff*scale')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 9
      CALL ef_set_arg_name(id, arg, '0')
      WRITE (buff, 60)
      CALL ef_set_arg_desc(id, arg, buff)
   60 FORMAT 
     .  ('Not used; included for backwards compatibility')
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
      SUBROUTINE scat2gridgauss_xy_work_size(id)

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
      INTEGER nxout, nyout, nx2, ny2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4)
      nyout = 1 + arg_hi_ss(Y_AXIS,ARG5) - arg_lo_ss(Y_AXIS,ARG5)

      nx2 = nxout* 2
      ny2 = nyout* 2

* xax  output x axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nx2, 1, 1, 1)

* yax  output y axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, ny2, 1, 1, 1)

* grid  work array - gridded data. 
      CALL ef_set_work_array_dims (id, 3, 
     .     arg_lo_ss(X_AXIS,ARG4), arg_lo_ss(Y_AXIS,ARG5), 1, 1, 
     .     arg_hi_ss(X_AXIS,ARG4), arg_hi_ss(Y_AXIS,ARG5), 1, 1)

* wate - weights.
      CALL ef_set_work_array_dims (id, 4, 
     .     arg_lo_ss(X_AXIS,ARG4), arg_lo_ss(Y_AXIS,ARG5), 1, 1, 
     .     arg_hi_ss(X_AXIS,ARG4), arg_hi_ss(Y_AXIS,ARG5), 1, 1)

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE scat2gridgauss_xy_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, result, 
     .     xax, yax, grid, wate)

*  arg_1  xpts \
*  arg_2  ypts  > Scattered x,y,f(x,y) triples to be gridded.
*  arg_3  fpts /    fpts can be fcn of z, t
*  arg_4  xaxis of new grid
*  arg_5  yaxis of new grid
*  arg_6  interpolation parameter   xscale
*  arg_7  interpolation parameter   yscale
*  arg_8  interpolation parameter   cutoff

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
     .      mem6lot:mem6hit)
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
      INTEGER i1, i2
      INTEGER i4, i4n, j5, j5n

      INTEGER nxpts, nypts, nscat
      INTEGER nx, ny, nxaxis, nyaxis
      INTEGER nxsize, nysize
      INTEGER i3, j3, k3, l3
      INTEGER i1n, i2n

      REAL x1, y1, xf, yf, t1, tf
      REAL xx, yy, tt
      REAL xsc, ysc, tsc
      REAL cutoff
      REAL val
      INTEGER iwflag

*  Dimension the work arrays

      REAL*8 xax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 yax(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL grid(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL wate(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)

      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulox(4), moduloy(4), regularx(4), 
     .        regulary(4)

      REAL dx, dy, xcut, ycut, xxbeg, xxend, yybeg, yyend


      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Get output axis characteristics

      CALL ef_get_axis_info (id, 4, ax_name, ax_units, backward, 
     .                       modulox, regularx)
      CALL ef_get_axis_info (id, 5, ax_name, ax_units, backward, 
     .                       moduloy, regulary)

*  Check that output axes are regular

      IF (.NOT. regularx(1)) then
         WRITE (errtxt, *) 'output axes must be REGULARLY spaced'
         GO TO 999
      ENDIF
      IF (.NOT. regulary(2)) then
         WRITE (errtxt, *) 'output axes must be REGULARLY spaced'
         GO TO 999
      ENDIF

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, Z, or T axis of the input arguments.

      nxpts = 0
      nypts = 0

      DO 100 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG1)
            i1n = arg_hi_ss(m,ARG1)
            if (i1n-i1 .NE. 0) nxpts = 1 + (i1n - i1)
         ENDIF
 100  CONTINUE

      DO 110 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG2)
            i2n = arg_hi_ss(m,ARG2)
            if (i2n-i2 .NE. 0) nypts = 1 + (i2n - i2)
         ENDIF
 110  CONTINUE

      IF (nxpts .NE. nypts .OR. nxpts .EQ. 0) GOTO 900

      nscat = nxpts

*  Compute number of points in output axes.

      i4 = arg_lo_ss(X_AXIS,ARG4)
      i4n = arg_hi_ss(X_AXIS,ARG4)

      j5 = arg_lo_ss(Y_AXIS,ARG5)
      j5n = arg_hi_ss(Y_AXIS,ARG5)

      nx = 1 + (i4n - i4)
      ny = 1 + (j5n - j5)
      nxsize = nx
      nysize = ny

*  Check that xax is a X axis and yax a Y axis

      IF (i4 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'fourth argument must be a X axis'
         GO TO 999
      ENDIF
      IF (j5 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'fifth argument must be a Y axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axes.

      call ef_get_coordinates(id, ARG4, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG4), arg_hi_ss(X_AXIS, ARG4), xax)

      call ef_get_coordinates(id, ARG5, Y_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG5), arg_hi_ss(Y_AXIS, ARG5), yax)

*  Set start, end, and delta for output axes. 

      x1 = xax(1,1,1,1)
      y1 = yax(1,1,1,1)

      xf = xax(nx,1,1,1)
      yf = yax(ny,1,1,1)

*  Time parameters for subroutine gausswt.  Calling with 1 time.

      nm = 1
      tt = 1.
      t1 = 1.
      tf = 1.
      tsc = 1. 

*  iwflag=1 for time wrapping; 0 for no wrapping

      iwflag = 0

*  Get interpolation parameters: mapping scales (data units)

      xsc  = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6), 
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6))

      ysc  = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7), 
     .            arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7))

      IF (xsc .LE. 0.) GOTO 910
      IF (ysc .LE. 0.) GOTO 920

*  And cutoff parameter:

      cutoff  = arg_8(arg_lo_ss(X_AXIS,ARG8), arg_lo_ss(Y_AXIS,ARG8), 
     .            arg_lo_ss(Z_AXIS,ARG8), arg_lo_ss(T_AXIS,ARG8))

      IF (cutoff .LE. 0.) GOTO 930

      nxaxis = arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4) + 1
      nyaxis = arg_hi_ss(Y_AXIS,ARG5) - arg_lo_ss(Y_AXIS,ARG5) + 1

*  Compute result at each time, and each Z

      i3 = arg_lo_ss(X_AXIS,ARG3)
      j3 = arg_lo_ss(Y_AXIS,ARG3)

      l3 = arg_lo_ss(T_AXIS,ARG3)
      DO 510 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      DO 500 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*  Initialize sums of values and weights.

         DO n = j5, j5n
            DO m = i4, i4n
               grid(m,n,1,1) = 0.
            ENDDO
         ENDDO

         DO n = j5, j5n
            DO m = i4, i4n
               wate(m,n,1,1) = 0.
            ENDDO
         ENDDO

*  Loop over x and y, compute the weighted sums for gaussian-weighted mapping
*  onto the grid.  Lat and longitude may be on the X,Y,Z or T axis of ARG1 and
*  ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

         DO 300 n = 1, nscat

            CALL pickoutxyz (arg_1, arg_2, n, xx, yy)

            IF (arg_lo_ss(X_AXIS,ARG3) .NE. ef_unspecified_int4) THEN
               val = arg_3(n,j3,k3,l3)
            ELSE IF(arg_lo_ss(Y_AXIS,ARG3) .NE. ef_unspecified_int4)THEN
               val = arg_3(i3,n,k3,l3)
            ENDIF

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.

            IF (modulox(1)) CALL modscat (xax, nx, 1, xx)
            IF (moduloy(2)) CALL modscat (yax, ny, 1, yy)

            IF ( val  .NE. bad_flag(ARG3) ) THEN
               CALL gausswt (xx, yy, tt, val, grid, wate, nx, ny, nm, 
     .                    x1, y1, t1, xf, yf, tf, xsc, ysc, tsc, 
     .                    cutoff, iwflag, nxaxis, nyaxis)


C ACM modulo 11/9/00  Put points within cutoff of the end just beyond the
C                     other end, and use in the gridding computation.

               IF (modulox(1)) THEN
                  dx = xf - x1
                  IF (nx .GT. 1) dx = (xf-x1)/real(nx-1)  ! gridbox size in data units
                  xcut = cutoff*xsc/dx		          ! cutoff scaled to grid units
                     
                  IF ((xx-x1 .GE. 0.) .AND. (xx-x1 .LT. xcut) ) THEN
                     xxend = xf + (xx-x1)
                     CALL gausswt(xxend, yy, tt, val, grid, wate, nx, 
     .                    ny, nm, x1, y1, t1, xf, yf, tf, xsc, ysc, tsc, 
     .                    cutoff, iwflag, nxaxis, nyaxis)
                  ENDIF
                     
                  IF ((xf-xx .GE. 0.)  .AND. (xf-xx .LT. xcut) ) THEN
                     xxbeg = x1 - (xf-xx)
                     CALL gausswt(xxbeg, yy, tt, val, grid, wate, nx, 
     .                    ny, nm, x1, y1, t1, xf, yf, tf, xsc, ysc, tsc, 
     .                    cutoff, iwflag, nxaxis, nyaxis)
                  ENDIF
               ENDIF

               IF (moduloy(2)) THEN
                  dy = yf - y1
                  IF (ny .GT. 1) dy = (yf-y1)/real(ny-1)  ! gridbox size in data units
                  ycut = cutoff*ysc/dy		          ! cutoff scaled to grid units

                  IF ((yy-y1 .GE. 0.0) .AND.  (yy-y1  .LT. ycut) ) THEN
                     yyend = yf + (yy-y1)
                     CALL gausswt(xx, yyend, tt, val, grid, wate, nx, 
     .                    ny, nm, x1, y1, t1, xf, yf, tf, xsc, ysc, tsc, 
     .                    cutoff, iwflag, nxaxis, nyaxis)
                  ENDIF
                     
                  IF ((yf-yy .LT. 0.)  .AND.  (yf-yy .LT. ycut) ) THEN
                     yybeg = y1 - (yf-yy)
                     CALL gausswt(xx, yybeg, tt, val, grid, wate, nx, 
     .                    ny, nm, x1, y1, t1, xf, yf, tf, xsc, ysc, tsc, 
     .                    cutoff, iwflag, nxaxis, nyaxis)
                  ENDIF
               ENDIF
            ENDIF


 300     CONTINUE

*  Put gridded fcn into result variable, dividing by summed weights. (as in
*  gaussfin, but indices needn't start at 1)

         DO 410 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         DO 400 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            result(i,j,k,l) = bad_flag_result
            IF ( wate(i,j,1,1) .gt. 0.) THEN
               result(i,j,k,l) = grid(i,j,1,1)/ wate(i,j,1,1)
            ELSE
               result(i,j,k,l) = bad_flag_result
            ENDIF

 400        CONTINUE
 410     CONTINUE

      k3 = k3 + arg_incr(Z_AXIS,ARG3)
 500  CONTINUE
      l3 = l3 + arg_incr(T_AXIS,ARG3)
 510  CONTINUE

      RETURN


900   CONTINUE

      IF (nxpts .NE. nypts) THEN
         WRITE (errtxt,20) nxpts, nypts
      ELSE IF (nxpts .EQ. 0) THEN
         WRITE (errtxt, 30)
      ENDIF
      GOTO 999 

 910  CONTINUE
      WRITE (errtxt,40) 'X', 6
      GOTO 999

 920  CONTINUE
      WRITE (errtxt,40) 'Y', 7
      GOTO 999

 930  CONTINUE
      write (errtxt, 50)  8
      GOTO 999


 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN


   20 FORMAT ('Input scattered x, y have different # of points', 2I8)

   30 FORMAT ('No data in scattered x, y points')

   40 FORMAT (a1,
     .    ' Mapping scale parameter must be positive. Argument', I2)

   50 FORMAT ('Cutoff parameter must be positive. Argument', I2)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      END
