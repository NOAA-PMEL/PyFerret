*
*  scat2grid_nobs_xyt.F
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
* July 2008  from scat2grid_nobs_xy

* Returns # observations in each grid cell when a variable is interpolated
* onto an equally-spaced X-Y-T grid.  Input is scattered locations xpts, ypts, tpts
* and the desired result grid.  Output is # observations in each grid cell
* when data is gridded in x and y. The time axis must have the same units and origin
* as the scattered time points. Calls routine "nobs" based on Billy
* Kessler's gaussmap-subs.f.
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


      SUBROUTINE  scat2grid_nobs_xyt_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .        'Count observations when gridding scattered ' //
     .        'data to XYT grid using Gaussian function')
      CALL ef_set_num_args(id, 6)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                NORMAL,          IMPLIED_BY_ARGS,
     .                                NORMAL,          NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

* Output grid is determined by arguments 4, 5, 6, the result's x, y, t axes.

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'X coordinates of scattered XYT locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Y coordinates of scattered XYT locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'T coordinates of scattered XYT locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'X axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, NO, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'YAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Y axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, NO, NO, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'TAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'T axis coordinates of a regular output grid, ' //
     .        'with same units and origin as TPTS')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, YES, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE scat2grid_nobs_xyt_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays, X/Y/Z/T/E/F dimensions
*
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER nxout, nyout, ntout, nx2, ny2, nt2

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4)
      nyout = 1 + arg_hi_ss(Y_AXIS,ARG5) - arg_lo_ss(Y_AXIS,ARG5)
      ntout = 1 + arg_hi_ss(T_AXIS,ARG6) - arg_lo_ss(T_AXIS,ARG6)

      nx2 = nxout * 2
      ny2 = nyout * 2
      nt2 = ntout * 2

* xax  output x axis
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                              nx2, 1, 1, 1, 1, 1)

* yax  output y axis
      CALL ef_set_work_array_dims_6d(id, 2,
     .                                1, 1, 1, 1, 1, 1,
     .                              ny2, 1, 1, 1, 1, 1)

* tax  output t axis
      CALL ef_set_work_array_dims_6d(id, 3,
     .                                1, 1, 1, 1, 1, 1,
     .                              nt2, 1, 1, 1, 1, 1)

* obs  work array - counts of observations
      CALL ef_set_work_array_dims_6d(id, 4,
     .                                1,     1,     1, 1, 1, 1,
     .                            nxout, nyout, ntout, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE  scat2grid_nobs_xyt_compute(id, arg_1, arg_2, arg_3,
     .                                    arg_4, arg_5, arg_6, result,
     .                                    xax, yax, tax, obs)

*  arg_1  xpts  \
*  arg_2  ypts   | Scattered x,y,t triplets to be gridded.
*  arg_2  tpts  /
*  arg_4  xaxis of new grid
*  arg_5  yaxis of new grid
*  arg_6  taxis of new grid

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz,
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz,
     .           mem3lot:mem3hit, mem3loe:mem3hie, mem3lof:mem3hif)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz,
     .           mem4lot:mem4hit, mem4loe:mem4hie, mem4lof:mem4hif)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz,
     .           mem5lot:mem5hit, mem5loe:mem5hie, mem5lof:mem5hif)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, mem6loz:mem6hiz,
     .           mem6lot:mem6hit, mem6loe:mem6hie, mem6lof:mem6hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* Ignore the unused final dimensions in the work arrays since
* Fortran is column major
      REAL*8 xax(wrk1lox:wrk1hix/2)
      REAL*8 yax(wrk2lox:wrk2hix/2)
      REAL*8 tax(wrk3lox:wrk3hix/2)
      REAL obs(wrk4lox:wrk4hix, wrk4loy:wrk4hiy, wrk4loz:wrk4hiz)

* After initialization, the 'res_' arrays contain indexing information
* for the result axes.  The 'arg_' arrays will contain the indexing
* information for each variable''s axes.

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l, m, n
      INTEGER idx, lo_ss, hi_ss
      INTEGER nxpts, nypts, ntpts, nscat
      INTEGER nx, ny, nt
      INTEGER i1, j1, l1
      REAL x1, y1, t1, xf, yf, tf
      REAL xx, yy, tt
      REAL dx, dy, dt, xxbeg, xxend, yybeg, yyend, ttbeg, ttend
      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), mdulo(6), regular(6)
      LOGICAL modulox, moduloy, modulot
      REAL*8 modxlen, modylen, modtlen

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Check to see if output axes are regular, and get modulo characteristics
      CALL ef_get_axis_info_6d(id, ARG4, ax_name, ax_units,
     .                         backward, mdulo, regular)
      IF ( .NOT. regular(X_AXIS) ) THEN
         errtxt = 'Output axes must be REGULARLY spaced'
         GOTO 999
      ENDIF
      modulox = mdulo(X_AXIS)
      IF ( modulox ) THEN
         CALL ef_get_axis_modulo_len(id, ARG4, X_AXIS, modxlen)
      ENDIF

      CALL ef_get_axis_info_6d(id, ARG5, ax_name, ax_units,
     .                         backward, mdulo, regular)
      IF ( .NOT. regular(Y_AXIS) ) THEN
         errtxt = 'Output axes must be REGULARLY spaced'
         GOTO 999
      ENDIF
      moduloy = mdulo(Y_AXIS)
      IF ( moduloy ) THEN
         CALL ef_get_axis_modulo_len(id, ARG5, Y_AXIS, modylen)
      ENDIF

      CALL ef_get_axis_info_6d(id, ARG6, ax_name, ax_units,
     .                         backward, mdulo, regular)
      IF ( .NOT. regular(T_AXIS) ) THEN
         errtxt = 'Output axes must be REGULARLY spaced'
         GOTO 999
      ENDIF
      modulot = mdulo(T_AXIS)
      IF ( modulot ) THEN
         CALL ef_get_axis_modulo_len(id, ARG6, T_AXIS, modtlen)
      ENDIF

*  Find number of points in scattered input points.  1-D arrays defining the
*  scattered data points may lie on the X, Y, Z, T, E, or F axis of the input
*  arguments.
*  Using as flattened arrays, so consider all axes.

      nxpts = 0
      DO 10 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG1)
         hi_ss = arg_hi_ss(idx,ARG1)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( nxpts .NE. 0 ) THEN
               nxpts = nxpts * (hi_ss - lo_ss + 1)
            ELSE
               nxpts = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  10  CONTINUE

      nypts = 0
      DO 20 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG2)
         hi_ss = arg_hi_ss(idx,ARG2)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( nypts .NE. 0 ) THEN
               nypts = nypts * (hi_ss - lo_ss + 1)
            ELSE
               nypts = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  20  CONTINUE

      ntpts = 0
      DO 40 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG3)
         hi_ss = arg_hi_ss(idx,ARG3)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( ntpts .NE. 0 ) THEN
               ntpts = ntpts * (hi_ss - lo_ss + 1)
            ELSE
               ntpts = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  40  CONTINUE

      IF ( nxpts .NE. nypts ) THEN
         WRITE (errtxt,90) ' XPTS,', ' YPTS ', nxpts, nypts
         GOTO 999
      ENDIF
      IF ( nxpts .NE. ntpts ) THEN
         WRITE (errtxt,90) ' XPTS,', ' TPTS ', nxpts, ntpts
         GOTO 999
      ENDIF
  90  FORMAT('Input scattered', 2A6, 'have different # of points', 2I8)

      IF ( nxpts .LE. 0 ) THEN
         errtxt = 'No data in scattered x, y, t points'
         GOTO 999
      ENDIF

      nscat = nxpts

*  Compute number of points in output axes.

      lo_ss = arg_lo_ss(X_AXIS,ARG4)
      hi_ss = arg_hi_ss(X_AXIS,ARG4)
*  Check that xax is an X axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fourth argument must be an X axis'
         GO TO 999
      ENDIF
      nx = hi_ss - lo_ss + 1

      lo_ss = arg_lo_ss(Y_AXIS,ARG5)
      hi_ss = arg_hi_ss(Y_AXIS,ARG5)
*  Check that yax a Y axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fifth argument must be a Y axis'
         GO TO 999
      ENDIF
      ny = hi_ss - lo_ss + 1

      lo_ss = arg_lo_ss(T_AXIS,ARG6)
      hi_ss = arg_hi_ss(T_AXIS,ARG6)
*  Check that tax a T axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Sixth argument must be a T axis'
         GO TO 999
      ENDIF
      nt = hi_ss - lo_ss + 1

C  Get coordinates of output axes.

      CALL ef_get_coordinates(id, ARG4, X_AXIS, arg_lo_ss(X_AXIS, ARG4),
     .                        arg_hi_ss(X_AXIS, ARG4), xax)

      CALL ef_get_coordinates(id, ARG5, Y_AXIS, arg_lo_ss(Y_AXIS, ARG5),
     .                        arg_hi_ss(Y_AXIS, ARG5), yax)

      CALL ef_get_coordinates(id, ARG6, T_AXIS, arg_lo_ss(T_AXIS, ARG6),
     .                        arg_hi_ss(T_AXIS, ARG6), tax)

*  Set start, end, and delta for output axes.

      x1 = xax(1)
      y1 = yax(1)
      t1 = tax(1)

      xf = xax(nx)
      yf = yax(ny)
      tf = tax(nt)

*  Gridbox sizes in data units
      dx = xf - x1
      IF ( nx .GT. 1 ) dx = (xf-x1) / REAL(nx-1)
      dy = yf - y1
      IF ( ny .GT. 1 ) dy = (yf-y1) / REAL(ny-1)
      dt = tf - t1
      IF ( nt .GT. 1 ) dt = (tf-t1) / REAL(nt-1)

*  Compute result

*  Initialize counts.

      DO l = 1, nt
         DO j = 1, ny
            DO i = 1, nx
               obs(i,j,l) = 0.
            ENDDO
         ENDDO
      ENDDO

*  Loop over x, y, and t, compute the number of observations for the gaussian-
*  weighted mapping onto the grid.  Lat, lon, and time may be on the X,Y,Z,T,E,
*  or F axis of ARG1 and ARG2, sending them to a subroutine collapses the extra
*  dimensions so the value can be found.

      DO 700 idx = 1, nscat

         CALL pickout3(arg_1, arg_2, arg_3, idx, xx, yy, tt)
         IF ( xx .EQ. bad_flag(ARG1) .OR.
     .        yy .EQ. bad_flag(ARG2) .OR.
     .        tt .EQ. bad_flag(ARG3) ) GOTO 700

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
         IF ( modulox ) THEN
            CALL modscat(xax(1), xax(nx), modxlen, 1, xx)
         ENDIF
         IF ( moduloy ) THEN
            CALL modscat(yax(1), yax(ny), modylen, 1, yy)
         ENDIF
         IF ( modulot ) THEN
            CALL modscat(tax(1), tax(nt), modtlen, 1, tt)
         ENDIF

         CALL nobsxyt(xx, yy, tt, nx, ny, nt, x1, y1, t1,
     .                dx, dy, dt, xf, yf, tf, obs)

*  Put points within dx or dy  or dt of the end just beyond the
*  other end, and use in the gridding computation.

         IF ( modulox ) THEN
            IF ( (xx-x1 .GE. 0.) .AND. (xx-x1 .LT. dx) ) THEN
               xxend = xf + (xx-x1)
               CALL nobsxyt(xxend, yy, tt, nx, ny, nt, x1, y1, t1,
     .                      dx, dy, dt, xf, yf, tf, obs)
            ENDIF

            IF ( (xf-xx .GE. 0.) .AND. (xf-xx .LT. dx) ) THEN
               xxbeg = x1 - (xf-xx)
               CALL nobsxyt(xxbeg, yy, tt, nx, ny, nt, x1, y1, t1,
     .                      dx, dy, dt, xf, yf, tf, obs)
            ENDIF
         ENDIF

         IF ( moduloy ) THEN
            IF ( (yy-y1 .GE. 0.) .AND. (yy-y1 .LT. dy) ) THEN
               yyend = yf + (yy-y1)
               CALL nobsxyt(xx, yyend, tt, nx, ny, nt, x1, y1, t1,
     .                      dx, dy, dt, xf, yf, tf, obs)
            ENDIF

            IF ( (yf-yy .LT. 0.) .AND. (yf-yy .LT. dy) ) THEN
               yybeg = y1 - (yf-yy)
               CALL nobsxyt(xx, yybeg, tt, nx, ny, nt, x1, y1, t1,
     .                      dx, dy, dt, xf, yf, tf, obs)
            ENDIF
         ENDIF

         IF ( modulot ) THEN
            IF ( (tt-t1 .GE. 0.) .AND. (tt-t1 .LT. dt) ) THEN
               ttend = tf + (tt-t1)
               CALL nobsxyt(xx, yy, ttend, nx, ny, nt, x1, y1, t1,
     .                      dx, dy, dt, xf, yf, tf, obs)
            ENDIF

            IF ( (tf-tt .LT. 0.) .AND. (tf-tt .LT. dt) ) THEN
               ttbeg = t1 - (tf-tt)
               CALL nobsxyt(xx, yy, ttbeg, nx, ny, nt, x1, y1, t1,
     .                      dx, dy, dt, xf, yf, tf, obs)
            ENDIF
         ENDIF

 700  CONTINUE

*  Put gridded fcn into result variable.
*  The result is normal to the Z, E, and F axes
      k = res_lo_ss(Z_AXIS)
      m = res_lo_ss(E_AXIS)
      n = res_lo_ss(F_AXIS)

      l1 = 1
      DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         j1 = 1
         DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            i1 = 1
            DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

               result(i,j,k,l,m,n) = obs(i1,j1,l1)

               i1 = i1 + 1
 100        CONTINUE

            j1 = j1 + 1
 200     CONTINUE

         l1 = l1 + 1
 400  CONTINUE

      RETURN

 999  CALL EF_BAIL_OUT(id, errtxt)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END
