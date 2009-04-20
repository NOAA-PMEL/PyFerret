*
*  scatgrid_bin_xyt.F
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
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
* Ansley Manke
* July 2008  from scatgrid_nobs_xyt

* Returns binned averages in each grid cell when a variable is interpolated 
* onto an equally-spaced X-Y-T grid.  Input is scattered locations xpts, ypts, tpts
* and the desired result grid.  Output is data binned and averaged in each grid cell 
* x, y, t. The time axis must have the same units and origin as the scattered time
* points. Calls routine "nobs" based on Billy Kessler's gaussmap-subs.f.
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


      SUBROUTINE  scatgrid_bin_xyt_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Computed binned averages of scattered ',
     .        'data onto a regularly-spaced XYT grid.')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 7)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS,
     .        IMPLIED_BY_ARGS, NORMAL, IMPLIED_BY_ARGS)

      CALL ef_set_num_work_arrays(id, 5)

* Output grid is determined by arguments 4, 5, 6, the result's x, y, t axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'X coordinates of scattered input pairs')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Y coordinates of scattered input pairs')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'T coordinates of scattered input pairs')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'VAR')
      CALL ef_set_arg_desc(id, arg, 
     .     'Variable to grid, at scattered XYT locations')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'X axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'YAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'Y axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, YES, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'TAXPTS')
      WRITE (buff, 20)
   20 FORMAT ('T axis coordinates, of a regular output grid, with same ',
     .       'units and origin as tpts')
      CALL ef_set_arg_desc(id, arg, buff)
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

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
      SUBROUTINE scatgrid_bin_xyt_work_size(id)

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
      INTEGER nxout, nyout, ntout, nx2, ny2, nt2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG5) - arg_lo_ss(X_AXIS,ARG5)
      nyout = 1 + arg_hi_ss(Y_AXIS,ARG6) - arg_lo_ss(Y_AXIS,ARG6)
      ntout = 1 + arg_hi_ss(T_AXIS,ARG7) - arg_lo_ss(T_AXIS,ARG7)

      nx2 = nxout* 2
      ny2 = nyout* 2
      nt2 = ntout* 2

* xax  output x axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nx2, 1, 1, 1)

* yax  output y axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, ny2, 1, 1, 1)

* Tax  output t axis
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, nt2, 1, 1, 1)

* obs  work array - gridded output
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, 
     .                             nxout, nyout, 1, ntout)
* counts  work array - gridded output
      CALL ef_set_work_array_dims (id, 5, 1, 1, 1, 1, 
     .                             nxout, nyout, 1, ntout)

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE  scatgrid_bin_xyt_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, arg_5, arg_6, arg_7, result, xax, yax, tax,  
     .     obs, counts)

*  arg_1  xpts  \
*  arg_2  ypts   | Scattered x,y,t locations to be gridded.
*  arg_3  tpts  / 
*  arg_4  var      Scattered data to be gridded
*  arg_5  xaxis of new grid   
*  arg_6  yaxis of new grid
*  arg_7  taxis of new grid

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .     mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .     mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .     mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, 
     .     mem4loz:mem4hiz, mem4lot:mem4hit)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, 
     .     mem5loz:mem5hiz, mem5lot:mem5hit)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, 
     .     mem6loz:mem6hiz, mem6lot:mem6hit)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, 
     .     mem7loz:mem7hiz, mem7lot:mem7hit)

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
      INTEGER i, j, k, l, m, n
      INTEGER i1, i2, i3, i4, j1, k1, l1
      INTEGER i1n, i2n, i3n, i4n
      INTEGER j4n, j5, j5n, L6, L6n

      INTEGER nxpts, nypts, ntpts, nscat
      INTEGER nx, ny, nt

      REAL x1, y1, t1, xf, yf, tf
      REAL xx, yy, tt, vv

*  Dimension the work arrays

      REAL*8 xax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 yax(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL*8 tax(wrk3lox:wrk3hix/2, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL obs(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)
      REAL counts(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .               wrk5loz:wrk5hiz, wrk5lot:wrk5hit)

      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulox(4), moduloy(4), modulot(4), regular(4)

      REAL dx, dy, xxbeg, xxend, yybeg, yyend

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Check to see if output axes are modulo

      CALL ef_get_axis_info (id, 5, ax_name, ax_units, backward, 
     .                       modulox, regular)
      CALL ef_get_axis_info (id, 6, ax_name, ax_units, backward, 
     .                       moduloy, regular)
      CALL ef_get_axis_info (id, 7, ax_name, ax_units, backward, 
     .                       modulot, regular)

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, Z, or T axis of the input arguments.

      nxpts = 0
      nypts = 0
      ntpts = 0
      nvpts = 0

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

      DO 120 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG3) .GE. 1) THEN
            i3 = arg_lo_ss(m,ARG3)
            i3n = arg_hi_ss(m,ARG3)
            if (i3n-i3 .NE. 0) ntpts = 1 + (i3n - i3)
         ENDIF
 120  CONTINUE

      DO 130 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG4) .GE. 1) THEN
            i4 = arg_lo_ss(m,ARG4)
            i4n = arg_hi_ss(m,ARG4)
            if (i4n-i4 .NE. 0) nvpts = 1 + (i4n - i4)
         ENDIF
 130  CONTINUE

      IF (nxpts .NE. nypts .OR. nxpts .EQ. 0) GOTO 900
      IF (nxpts .NE. ntpts) GOTO 900
      IF (nxpts .NE. nvpts) GOTO 900

      nscat = nxpts

*  Compute number of points in output axes.

      i5 = ef_unspecified_int4
      j6 = ef_unspecified_int4
      L7 = ef_unspecified_int4

      i5 = arg_lo_ss(X_AXIS,ARG5)
      i5n = arg_hi_ss(X_AXIS,ARG5)

      j6 = arg_lo_ss(Y_AXIS,ARG6)
      j6n = arg_hi_ss(Y_AXIS,ARG6)

      L7 = arg_lo_ss(T_AXIS,ARG7)
      L7n = arg_hi_ss(T_AXIS,ARG7)

      nx = 1 + (i5n - i5)
      ny = 1 + (j6n - j6)
      nt = 1 + (L7n - L7)

*  Check that xax is a X axis and yax a Y axis

      IF (i5.EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Fifth argument must be a X axis'
         GO TO 999
      ENDIF
      IF (j6 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Sixth argument must be a Y axis'
         GO TO 999
      ENDIF
      IF (L7 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Seventh argument must be a T axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axes.

      call ef_get_coordinates(id, ARG5, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG5), arg_hi_ss(X_AXIS, ARG5), xax)
      print *, ' xax ', (xax(i,1,1,1), i=1,5)

      call ef_get_coordinates(id, ARG6, Y_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG6), arg_hi_ss(Y_AXIS, ARG6), yax)
      print *, ' yax ', (yax(i,1,1,1), i=1,5)

      call ef_get_coordinates(id, ARG7, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG7), arg_hi_ss(T_AXIS, ARG7), tax)
      print *, ' tax ', (tax(i,1,1,1), i=1,5)

*  Set start, end, and delta for output axes. 

      x1 = xax(1,1,1,1)
      y1 = yax(1,1,1,1)
      t1 = tax(1,1,1,1)

      xf = xax(nx,1,1,1)
      yf = yax(ny,1,1,1)
      tf = tax(nt,1,1,1)

*  Gridbox sizes in data units
      dx = xf - x1
      IF (nx .GT. 1) dx = (xf-x1)/ REAL(nx-1)
      dy = yf - y1
      IF (ny .GT. 1) dy = (yf-y1)/ REAL(ny-1) 
      dt = tf - t1
      IF (nt .GT. 1) dt = (tf-t1)/ REAL(nt-1) 

*  Compute result

*  Initialize counts and sums of observations

      DO L = 1, nt
         DO j = 1, ny
            DO i = 1, nx
               counts(i,j,1,L) = 0.
            ENDDO
         ENDDO
      ENDDO

      DO L = 1, nt
         DO j = 1, ny
            DO i = 1, nx
               obs(i,j,1,L) = 0.
            ENDDO
         ENDDO
      ENDDO

*  Loop over x and y, compute the number of observations for the gaussian-weighted 
*  mapping onto the grid.  Lat and longitude may be on the X,Y,Z or T axis of ARG1 
*  and ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

         DO 300 n = 1, nscat

            CALL pickoutxyztv (arg_1, arg_2, arg_3, arg_4, n, xx, yy, tt, vv)

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.

            IF (modulox(1)) CALL modscat (xax, nx, 1, xx)
            IF (moduloy(2)) CALL modscat (yax, ny, 1, yy)
            IF (modulot(4)) CALL modscat (tax, nt, 1, tt)

	    CALL binxyt (xx, yy, tt, vv, nx, ny, nt, x1, y1, t1, 
     .                 dx, dy, dt, xf, yf, tf, obs, counts)

*  Put points within dx or dy  or dt of the end just beyond the
*  other end, and use in the gridding computation.

            IF (modulox(1)) THEN
               IF ((xx-x1 .GE. 0.) .AND. (xx-x1 .LT. dx) ) THEN
                  xxend = xf + (xx-x1)
                  CALL binxyt (xxend, yy, tt, vv, nx, ny, nt, 
     .               x1, y1, t1, dx, dy, dt, xf, yf, tf, obs, counts)
               ENDIF
                     
               IF ((xf-xx .GE. 0.)  .AND. (xf-xx .LT. dx) ) THEN
                  xxbeg = x1 - (xf-xx)
                  CALL binxyt (xxbeg, yy, tt, vv, nx, ny, nt, 
     .               x1, y1, t1, dx, dy, dt,xf, yf, tf, obs, counts)
               ENDIF
            ENDIF

            IF (modulot(4)) THEN
               IF ((tt-t1 .GE. 0.0) .AND.  (tt-t1  .LT. dt) ) THEN
                  ttend = tf + (tt-t1)
                  CALL binxyt (xx, yy, ttend, vv, nx, ny, nt, 
     .               x1, y1, t1, dx, dy, dt,xf, yf, tf, obs, counts)
               ENDIF
                     
               IF ((tf-tt .LT. 0.)  .AND.  (tf-tt .LT. dt) ) THEN
                  ttbeg = t1 - (tf-tt)
                  CALL binxyt (xx, yy, ttbeg, vv, nx, ny, nt, 
     .               x1, y1, t1, dx, dy, dt,xf, yf, tf, obs, counts)
               ENDIF
            ENDIF


 300     CONTINUE

*  Put gridded fcn into result variable.

         k = res_lo_ss(Z_AXIS)

         i1 = 1
         DO 420 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
            j1 = 1
            DO 410 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
               L1 = 1
               DO 400 L = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                  IF (counts(i1,j1,1,L1) .GT. 0.) THEN
                     result(i,j,k,l) = obs(i1,j1,1,L1)/counts(i1,j1,1,L1)
                  ELSE
                     result(i,j,k,l) = bad_flag_result
                  ENDIF

                  L1 = L1 + 1
 400           CONTINUE
               j1 = j1 + 1

 410        CONTINUE
            i1 = i1 + 1
 420     CONTINUE

      RETURN

900   CONTINUE

      IF (nxpts .NE. nypts) THEN
         WRITE (errtxt,20) nxpts, nypts, ntpts, nvpts
      ELSE IF (nxpts .NE. ntpts) THEN
         WRITE (errtxt,20) nxpts, nypts, ntpts, nvpts
      ELSE IF (nxpts .NE. nvpts) THEN
         WRITE (errtxt,20) nxpts, nypts, ntpts, nvpts
      ELSE IF (nxpts .EQ. 0) THEN
         WRITE (errtxt, 30)
      ENDIF
      GOTO 999 

 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN


   20 FORMAT ('Input scattered x,y,t,v have different # of points', 4I8)

   30 FORMAT ('No data in scattered x,y,t,v points')


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      END




      SUBROUTINE pickoutxyztv (alon, alat, atimes, vals, n, xx, yy, tt, vv)

*  Called from the scatter*  gridding functins.
*  Pick out nth item from alon and alat, which are really 4-dimensioned arrays
*  with normal axes on three of the four axes.

      REAL alon(*), alat(*), atimes(*), vals(*)
      REAL xx, yy, tt, vv
      INTEGER n

      xx = alon(n)
      yy = alat(n)
      tt = atimes(n)
      vv = vals(n)

      RETURN
      END



c************************************************************************
c...binxyt.....Based on gausswt, return binned/averaged obs in (x,y) gridbox
c............method is to call this sub for each data value
c............all calcs done in gridbox units
c............xx/x1/xf all in same units
c............yy/y1/yf all in same units

c  i	xx,yy,tt=x/y/t location of data pt (data units)
c  i	nx,ny,nt=size of grids
c  i	x1,y1,L1=west/south/earliest edge of grid (center of 1st box in data units)
c  o 	obs(nx,ny,nt) number of obs in each (x,y,t) gridbox
c--------------------------------------------------------------------------

	SUBROUTINE binxyt (xx, yy, tt, vv, nx, ny, nt, x1, y1, t1, 
     .                 dx, dy, dt, xf, yf, tf, obs, counts)

* Argument declarations

        INTEGER nx, ny, nt
	REAL xx, yy, tt, vv, x1, y1, t1, dx, dy, dt, xf, 
     .       yf, tf, obs(nx,ny,nt), counts(nx,ny,nt)

* Local variable declarations

        REAL xxg, yyg, ttg
        INTEGER ig, jg, Lg 

	xxg = xx
	yyg = yy
	ttg = tt
	IF (dx .GT. 0.) xxg = (xx-x1)/ dx + 1.   ! grid values of data location
	IF (dy .GT. 0.) yyg = (yy-y1)/ dy + 1.
	IF (dt .GT. 0.) ttg = (tt-t1)/ dt + 1.

c.................save the number of obs in each (x,y,t) gridbox
	ig = NINT(xxg)
	jg = NINT(yyg)
	Lg = NINT(ttg)
	IF (ig.GE.1 .AND. ig.LE.nx .AND. jg.GE.1 .AND. jg.LE.ny .AND.
     .      Lg.GE.1 .AND. Lg.LE.nt   ) THEN
           counts(ig,jg,Lg) = counts(ig,jg,Lg)+1.
           obs(ig,jg,Lg) = obs(ig,jg,Lg) + vv
        ENDIF

	RETURN
	END



