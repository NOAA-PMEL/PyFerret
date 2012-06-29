*
*  scat2grid_nobs_xy.F
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
* February 9, 2005 from scat2gridgauss_xy

* Returns # observations in each grid cell when a variable is interpolated
* onto an equally-spaced X-Y grid.  Input is scattered locations xpts, ypts
* and the desired result grid. Output is # observations in each grid cell
* when data is gridded in x and y.  Calls routine "nobs" based on Billy
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


      SUBROUTINE  scat2grid_nobs_xy_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .        'Count observations when gridding scattered ' //
     .        'data to XY grid using Gaussian function')
      CALL ef_set_num_args(id, 4)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                NORMAL,          NORMAL,
     .                                NORMAL,          NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 3)

* Output grid is determined by arguments 3 and 4, the result's x and y axes.

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'X coordinates of scattered XY locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Y coordinates of scattered XY locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'X axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'YAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Y axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, NO, NO, NO, NO)

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
      SUBROUTINE scat2grid_nobs_xy_work_size(id)

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
      INTEGER nxout, nyout, nx2, ny2

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3)
      nyout = 1 + arg_hi_ss(Y_AXIS,ARG4) - arg_lo_ss(Y_AXIS,ARG4)

      nx2 = nxout * 2
      ny2 = nyout * 2

* xax  output x axis
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                              nx2, 1, 1, 1, 1, 1)

* yax  output y axis
      CALL ef_set_work_array_dims_6d(id, 2,
     .                                1, 1, 1, 1, 1, 1,
     .                              ny2, 1, 1, 1, 1, 1)

* obs  work array - counts of observations
      CALL ef_set_work_array_dims_6d(id, 3,
     .                                1,     1, 1, 1, 1, 1,
     .                            nxout, nyout, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE scat2grid_nobs_xy_compute(id, arg_1, arg_2, arg_3,
     .                                  arg_4, result, xax, yax, obs)

*  arg_1  xpts  \  Scattered x,y pairs to be gridded.
*  arg_2  ypts  /
*  arg_3  xaxis of new grid
*  arg_4  yaxis of new grid

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

      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* Ignore the unused final dimensions in the work arrays since
* Fortran is column major
      REAL*8 xax(wrk1lox:wrk1hix/2)
      REAL*8 yax(wrk2lox:wrk2hix/2)
      REAL obs(wrk3lox:wrk3hix, wrk3loy:wrk3hiy)

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
      INTEGER nxpts, nypts, nscat
      INTEGER nx, ny
      INTEGER i1, j1
      REAL x1, y1, xf, yf
      REAL xx, yy
      REAL dx, dy, xxbeg, xxend, yybeg, yyend
      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), mdulo(6), regular(6)
      LOGICAL modulox, moduloy
      REAL*8 modxlen, modylen

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Check to see if output axes are regular, and get modulo characteristics
      CALL ef_get_axis_info_6d(id, ARG3, ax_name, ax_units,
     .                         backward, mdulo, regular)
      IF ( .NOT. regular(X_AXIS) ) THEN
         errtxt = 'Output axes must be REGULARLY spaced'
         GOTO 999
      ENDIF
      modulox = mdulo(X_AXIS)
      IF ( modulox ) THEN
         CALL ef_get_axis_modulo_len(id, ARG3, X_AXIS, modxlen)
      ENDIF

      CALL ef_get_axis_info_6d(id, ARG4, ax_name, ax_units,
     .                         backward, mdulo, regular)
      IF ( .NOT. regular(Y_AXIS) ) THEN
         errtxt = 'Output axes must be REGULARLY spaced'
         GOTO 999
      ENDIF
      moduloy = mdulo(Y_AXIS)
      IF ( moduloy ) THEN
         CALL ef_get_axis_modulo_len(id, ARG4, Y_AXIS, modylen)
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

      IF ( nxpts .NE. nypts ) THEN
         WRITE (errtxt,90) ' XPTS,', ' YPTS ', nxpts, nypts
         GOTO 999
      ENDIF
  90  FORMAT('Input scattered', 2A6, 'have different # of points', 2I8)

      IF ( nxpts .LE. 0 ) THEN
         errtxt = 'No data in scattered x, y points'
         GOTO 999
      ENDIF

      nscat = nxpts

*  Compute number of points in output axes.

      lo_ss = arg_lo_ss(X_AXIS,ARG3)
      hi_ss = arg_hi_ss(X_AXIS,ARG3)
*  Check that xax is an X axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Third argument must be an X axis'
         GO TO 999
      ENDIF
      nx = hi_ss - lo_ss + 1

      lo_ss = arg_lo_ss(Y_AXIS,ARG4)
      hi_ss = arg_hi_ss(Y_AXIS,ARG4)
*  Check that yax a Y axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fourth argument must be a Y axis'
         GO TO 999
      ENDIF
      ny = hi_ss - lo_ss + 1

C  Get coordinates of output axes.

      CALL ef_get_coordinates(id, ARG3, X_AXIS, arg_lo_ss(X_AXIS, ARG3),
     .                        arg_hi_ss(X_AXIS, ARG3), xax)

      CALL ef_get_coordinates(id, ARG4, Y_AXIS, arg_lo_ss(Y_AXIS, ARG4),
     .                        arg_hi_ss(Y_AXIS, ARG4), yax)

*  Set start, end, and delta for output axes.

      x1 = xax(1)
      y1 = yax(1)

      xf = xax(nx)
      yf = yax(ny)

*  Gridbox sizes in data units
      dx = xf - x1
      IF ( nx .GT. 1 ) dx = (xf-x1) / REAL(nx-1)
      dy = yf - y1
      IF ( ny .GT. 1 ) dy = (yf-y1) / REAL(ny-1)

*  Compute result

*  Initialize counts.

      DO j = 1, ny
         DO i = 1, nx
            obs(i,j) = 0.
         ENDDO
      ENDDO

*  Loop over x and y, compute the number of observations for the gaussian-weighted
*  mapping onto the grid.  Lat and longitude may be on the X,Y,Z,T,E, or F axis of
*  ARG1 and ARG2, sending them to a subroutine collapses the extra dimensions so
*  the value can be found.

      DO 700 idx = 1, nscat

         CALL pickout2(arg_1, arg_2, idx, xx, yy)
         IF ( xx .EQ. bad_flag(ARG1) .OR.
     .        yy .EQ. bad_flag(ARG2) ) GOTO 700

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
         IF ( modulox ) THEN
            CALL modscat(xax(1), xax(nx), modxlen, 1, xx)
         ENDIF
         IF ( moduloy ) THEN
            CALL modscat(yax(1), yax(ny), modylen, 1, yy)
         ENDIF

         CALL nobs(xx, yy, nx, ny, x1, y1, dx, dy, xf, yf, obs)

C ACM modulo 11/9/00  Put points within dx or dy of the end just beyond the
C                     other end, and use in the gridding computation.

         IF ( modulox ) THEN
            IF ( (xx-x1 .GE. 0.) .AND. (xx-x1 .LT. dx) ) THEN
               xxend = xf + (xx-x1)
               CALL nobs(xxend, yy, nx, ny, x1, y1, dx, dy, xf, yf, obs)
            ENDIF

            IF ( (xf-xx .GE. 0.) .AND. (xf-xx .LT. dx) ) THEN
               xxbeg = x1 - (xf-xx)
               CALL nobs(xxbeg, yy, nx, ny, x1, y1, dx, dy, xf, yf, obs)
            ENDIF
         ENDIF

         IF ( moduloy ) THEN
            IF ( (yy-y1 .GE. 0.) .AND. (yy-y1 .LT. dy) ) THEN
               yyend = yf + (yy-y1)
               CALL nobs(xx, yyend, nx, ny, x1, y1, dx, dy, xf, yf, obs)
            ENDIF

            IF ( (yf-yy .LT. 0.) .AND. (yf-yy .LT. dy) ) THEN
               yybeg = y1 - (yf-yy)
               CALL nobs(xx, yybeg, nx, ny, x1, y1, dx, dy, xf, yf, obs)
            ENDIF
         ENDIF

 700  CONTINUE

*  Put gridded fcn into result variable.
*  The result is normal to the Z, T, E, and F axes
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)
      m = res_lo_ss(E_AXIS)
      n = res_lo_ss(F_AXIS)

      j1 = 1
      DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         i1 = 1
         DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            result(i,j,k,l,m,n) = obs(i1,j1)

            i1 = i1 + 1
 100     CONTINUE

         j1 = j1 + 1
 200  CONTINUE

      RETURN

 999  CALL EF_BAIL_OUT(id, errtxt)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END
