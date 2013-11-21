*
*  scat2gridgauss_yt_v0
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
* Nov 7, 2002 from scat2gridgauss_xy
c V62 *ACM* 2/2009 Call gausswt2_v0, with the original treatment of the
c            cutoff parameter, for backwards compatibility.

* Returns variable interpolated onto an equally-spaced Y-T grid.
* Input is scattered triples: (y, t, variable) , f may be function of x, z, e, f
* Output is gridded data in y, t.  Calls routine "gausswt2_v0".
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


      SUBROUTINE  scat2gridgauss_yt_v0_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 'Use Gaussian v0 weighting to grid ' //
     .                     'scattered data to a YT grid.')
      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

* YT grid is determined by arguments 4 and 5, the result's y and t axes.

      arg = 1
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Y coordinates of scattered YT locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'T coordinates of scattered YT locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      CALL ef_set_arg_desc(id, arg,
     .        'variable at scattered YT locations. ' //
     .        'May also be fcn of X,Z,E,F')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, NO, YES, NO, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'YAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Y axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'TAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'T axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, YES, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'YSCALE')
      CALL ef_set_arg_desc(id, arg,
     .  'Mapping scale in Y direction, in data units (e.g. km or lat)')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'TSCALE')
      CALL ef_set_arg_desc(id, arg,
     .  'Mapping scale in T direction, in data units (e.g. hrs, days)')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 8
      CALL ef_set_arg_name(id, arg, 'CUTOFF')
      CALL ef_set_arg_desc(id, arg,
     .        'Weight cutoff: use pts within cutoff*scale')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 9
      CALL ef_set_arg_name(id, arg, '0')
      CALL ef_set_arg_desc(id, arg,
     .        'Not used; included for backwards compatibility')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

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
      SUBROUTINE scat2gridgauss_yt_v0_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
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
      INTEGER nyout, ntout, ny2, nt2

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nyout = 1 + arg_hi_ss(Y_AXIS,ARG4) - arg_lo_ss(Y_AXIS,ARG4)
      ntout = 1 + arg_hi_ss(T_AXIS,ARG5) - arg_lo_ss(T_AXIS,ARG5)

      ny2 = nyout * 2
      nt2 = ntout * 2

* yax  output y axis
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                              ny2, 1, 1, 1, 1, 1)

* tax  output t axis
      CALL ef_set_work_array_dims_6d(id, 2,
     .                                1, 1, 1, 1, 1, 1,
     .                              nt2, 1, 1, 1, 1, 1)

* grid  work array - gridded data.
      CALL ef_set_work_array_dims_6d(id, 3,
     .                                1,     1, 1, 1, 1, 1,
     .                            nyout, ntout, 1, 1, 1, 1)

* wate - weights.
      CALL ef_set_work_array_dims_6d(id, 4,
     .                                1,     1, 1, 1, 1, 1,
     .                            nyout, ntout, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE scat2gridgauss_yt_v0_compute(id, arg_1, arg_2, arg_3,
     .                           arg_4, arg_5, arg_6, arg_7, arg_8,
     .                           arg_9, result, yax, tax, grid, wate)

*  arg_1  ypts \
*  arg_2  tpts  > Scattered y,t,variable,  triples to be gridded.
*  arg_3  variable, /    variable, can be fcn of x,z,e,f
*  arg_4  yaxis of output grid
*  arg_5  taxis of output grid
*  arg_6  interpolation parameter   yscale
*  arg_7  interpolation parameter   tscale
*  arg_8  interpolation parameter   cutoff
*  arg_9  not used

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
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, mem7loz:mem7hiz,
     .           mem7lot:mem7hit, mem7loe:mem7hie, mem7lof:mem7hif)
      REAL arg_8(mem8lox:mem8hix, mem8loy:mem8hiy, mem8loz:mem8hiz,
     .           mem8lot:mem8hit, mem8loe:mem8hie, mem8lof:mem8hif)
      REAL arg_9(mem9lox:mem9hix, mem9loy:mem9hiy, mem9loz:mem9hiz,
     .           mem9lot:mem9hit, mem9loe:mem9hie, mem9lof:mem9hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* Ignore the unused final dimensions in the work arrays since
* Fortran is column major
      REAL*8 yax(wrk1lox:wrk1hix/2)
      REAL*8 tax(wrk2lox:wrk2hix/2)
      REAL grid(wrk3lox:wrk3hix, wrk3loy:wrk3hiy)
      REAL wate(wrk4lox:wrk4hix, wrk4loy:wrk4hiy)

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

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l, m, n
      INTEGER idx, lo_ss, hi_ss
      INTEGER nypts, ntpts, nscat
      INTEGER nx, ny, nt
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER ylo, yhi, tlo, thi
      REAL x1, y1, t1, xf, yf, tf
      REAL xx, yy, tt
      REAL xsc, ysc, tsc
      REAL cutoff
      REAL val
      REAL dy, dt, ycut, tcut, yybeg, yyend, ttbeg, ttend
      INTEGER iwflag
      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), mdulo(6), regular(6)
      LOGICAL moduloy, modulot
      REAL*8 modylen, modtlen

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Check to see if output axes are regular, and get modulo characteristics
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

      CALL ef_get_axis_info_6d(id, ARG5, ax_name, ax_units,
     .                         backward, mdulo, regular)
      IF ( .NOT. regular(T_AXIS) ) THEN
         errtxt = 'Output axes must be REGULARLY spaced'
         GOTO 999
      ENDIF
      modulot = mdulo(T_AXIS)
      IF ( modulot ) THEN
         CALL ef_get_axis_modulo_len(id, ARG5, T_AXIS, modtlen)
      ENDIF

*  Find number of points in scattered input points.  1-D arrays defining the
*  scattered data points may lie on the X, Y, Z, T, E, or f axis of the input
*  arguments.
*  Using as flattened arrays, so consider all axes.

      nypts = 0
      DO 20 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG1)
         hi_ss = arg_hi_ss(idx,ARG1)
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
         lo_ss = arg_lo_ss(idx,ARG2)
         hi_ss = arg_hi_ss(idx,ARG2)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( ntpts .NE. 0 ) THEN
               ntpts = ntpts * (hi_ss - lo_ss + 1)
            ELSE
               ntpts = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  40  CONTINUE

      IF ( nypts .NE. ntpts ) THEN
         WRITE (errtxt,90) ' YPTS,', ' TPTS ', nypts, ntpts
         GOTO 999
      ENDIF

* Verify the number of values given in ARG3 makes sense
      ylo = arg_lo_ss(Y_AXIS,ARG3)
      yhi = arg_hi_ss(Y_AXIS,ARG3)
      tlo = arg_lo_ss(T_AXIS,ARG3)
      thi = arg_hi_ss(T_AXIS,ARG3)

      IF ( (yhi .GT. ylo) .AND. (thi .GT. tlo) ) THEN
         errtxt = 'F() data values defined on both Y and T axis'
         GOTO 999
      ELSE IF ( yhi .GT. ylo ) THEN
         nscat = yhi - ylo + 1
         IF ( nypts .NE. nscat ) THEN
            WRITE (errtxt,90) ' YPTS,', ' F(Y) ', nypts, nscat
            GOTO 999
         ENDIF
      ELSE IF ( thi .GT. tlo ) THEN
         nscat = thi - tlo + 1
         IF ( nypts .NE. nscat ) THEN
            WRITE (errtxt,90) ' YPTS,', ' F(T) ', nypts, nscat
            GOTO 999
         ENDIF
      ELSE IF ( (ylo .NE. ef_unspecified_int4) .OR.
     .          (tlo .NE. ef_unspecified_int4) ) THEN
         nscat = 1
         IF ( nypts .NE. nscat ) THEN
            WRITE (errtxt,90) ' YPTS,', ' F()  ', nypts, nscat
            GOTO 999
         ENDIF
      ELSE
         errtxt = 'No data in scattered y, t points.  ' //
     .            'F() data values must be defined on Y or T axis.'
         GOTO 999
      ENDIF
  90  FORMAT('Input scattered', 2A6, 'have different # of points', 2I8)

*  Compute number of points in output axes.

      lo_ss = arg_lo_ss(Y_AXIS,ARG4)
      hi_ss = arg_hi_ss(Y_AXIS,ARG4)
*  Check that yax is a Y axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fourth argument must be a Y axis'
         GO TO 999
      ENDIF
      ny = hi_ss - lo_ss + 1

      lo_ss = arg_lo_ss(T_AXIS,ARG5)
      hi_ss = arg_hi_ss(T_AXIS,ARG5)
*  Check that tax a T axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fifth argument must be a T axis'
         GO TO 999
      ENDIF
      nt = hi_ss - lo_ss + 1

C  Get coordinates of output axes.

      CALL ef_get_coordinates(id, ARG4, Y_AXIS, arg_lo_ss(Y_AXIS, ARG4),
     .                        arg_hi_ss(Y_AXIS, ARG4), yax)

      CALL ef_get_coordinates(id, ARG5, T_AXIS, arg_lo_ss(T_AXIS, ARG5),
     .                        arg_hi_ss(T_AXIS, ARG5), tax)

*  Set start and end for output axes.

      y1 = yax(1)
      t1 = tax(1)

      yf = yax(ny)
      tf = tax(nt)

*  X parameters for subroutine gausswt.  Calling with 1 x value

      nx = 1
      xx = 1.
      x1 = 1.
      xf = 1.
      xsc = 1.

*  iwflag=1 for time wrapping; 0 for no wrapping

      iwflag = 0

*  Get interpolation parameters: mapping scales (data units)

      ysc = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6),
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6),
     .            arg_lo_ss(E_AXIS,ARG6), arg_lo_ss(F_AXIS,ARG6))
      IF ( ysc .LE. 0. ) THEN
         errtxt = 'Y mapping scale parameter must be positive'
         GOTO 999
      ENDIF

      tsc = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7),
     .            arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7),
     .            arg_lo_ss(E_AXIS,ARG7), arg_lo_ss(F_AXIS,ARG7))
      IF ( tsc .LE. 0. ) THEN
         errtxt = 'T mapping scale parameter must be positive'
         GOTO 999
      ENDIF

*  And cutoff parameter:

      cutoff = arg_8(arg_lo_ss(X_AXIS,ARG8), arg_lo_ss(Y_AXIS,ARG8),
     .               arg_lo_ss(Z_AXIS,ARG8), arg_lo_ss(T_AXIS,ARG8),
     .               arg_lo_ss(E_AXIS,ARG8), arg_lo_ss(F_AXIS,ARG8))

      IF ( cutoff .LE. 0. ) THEN
         errtxt = 'Cutoff parameter must be positive'
         GOTO 999
      ENDIF

*  Compute result at each X, Z, E, F

      n3 = arg_lo_ss(F_AXIS,ARG3)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m3 = arg_lo_ss(E_AXIS,ARG3)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      k3 = arg_lo_ss(Z_AXIS,ARG3)
      DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

      i3 = arg_lo_ss(X_AXIS,ARG3)
      DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

*  Initialize sums of values and weights.

         DO l3 = 1, nt
            DO j3 = 1, ny
               grid(j3,l3) = 0.
            ENDDO
         ENDDO

         DO l3 = 1, nt
            DO j3 = 1, ny
               wate(j3,l3) = 0.
            ENDDO
         ENDDO

*  Loop over y and t, compute the weighted sums for gaussian-weighted mapping
*  onto the grid.  Args 1-2 may be on the X,Y,Z,T,E or F axis of ARG1 and
*  ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

         DO 700 idx = 1, nscat

            CALL pickout2(arg_1, arg_2, idx, yy, tt)
            IF ( yhi .GT. ylo ) THEN
               j3 = ylo + idx - 1
               l3 = tlo
            ELSE
               j3 = ylo
               l3 = tlo + idx - 1
            ENDIF
            val = arg_3(i3,j3,k3,l3,m3,n3)

            IF (  yy .EQ. bad_flag(ARG1) .OR.
     .            tt .EQ. bad_flag(ARG2) .OR.
     .           val .EQ. bad_flag(ARG3) ) GOTO 700

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
            IF ( moduloy ) THEN
               CALL modscat(yax(1), yax(ny), modylen, 1, yy)
            ENDIF
            IF ( modulot ) THEN
               CALL modscat(tax(1), tax(nt), modtlen, 1, tt)
            ENDIF

            CALL gausswt2_v0(yy, tt, xx, val, grid, wate, ny,
     .                    nt, nx, y1, t1, x1, yf, tf, xf, ysc,
     .                    tsc, xsc, cutoff, iwflag, ny, nt)

C ACM modulo 11/9/00  Put points within cutoff of the end just beyond the
C                     other end, and use in the gridding computation.

            IF ( moduloy ) THEN
               dy = yf - y1
               IF (ny .GT. 1) dy = (yf-y1)/real(ny-1)  ! gridbox size in data units
               ycut = cutoff*ysc/dy                    ! cutoff scaled to grid units

               IF ( (yy-y1 .GE. 0.) .AND. (yy-y1 .LT. ycut) ) THEN
                  yyend = yf + (yy-y1)
                  CALL gausswt2_v0(yyend, tt, xx, val, grid, wate, ny,
     .                          nt, nx, y1, t1, x1, yf, tf, xf, ysc,
     .                          tsc, ysc, cutoff, iwflag, ny, nt)
               ENDIF

               IF ( (yf-yy .GE. 0.) .AND. (yf-yy .LT. ycut) ) THEN
                  yybeg = y1 - (yf-yy)
                  CALL gausswt2_v0(yybeg, tt, xx, val, grid, wate, ny,
     .                          nt, nx, y1, t1, x1, yf, tf, xf, ysc,
     .                          tsc, xsc, cutoff, iwflag, ny, nt)
               ENDIF
            ENDIF

            IF ( modulot ) THEN
               dt = tf - t1
               IF (nt .GT. 1) dt = (tf-t1)/real(nt-1)  ! gridbox size in data units
               tcut = cutoff*tsc/dt                    ! cutoff scaled to grid units

               IF ( (tt-t1 .GE. 0.) .AND. (tt-t1 .LT. tcut) ) THEN
                  ttend = tf + (tt-t1)
                  CALL gausswt2_v0(yy, ttend, xx, val, grid, wate, ny,
     .                          nt, nx, y1, t1, x1, yf, tf, xf, ysc,
     .                          tsc, xsc, cutoff, iwflag, ny, nt)
               ENDIF

               IF ( (tf-tt .LT. 0.) .AND. (tf-tt .LT. tcut) ) THEN
                  ttbeg = y1 - (yf-tt)
                  CALL gausswt2_v0(yy, ttbeg, xx, val, grid, wate, ny,
     .                          nt, nx, y1, t1, x1, yf, tf, xf, ysc,
     .                          tsc, xsc, cutoff, iwflag, ny, nt)
               ENDIF
            ENDIF

 700     CONTINUE

*  Put gridded fcn into result variable, dividing by summed weights. (as in
*  gaussfin, but indices needn't start at 1)

         l3 = 1
         DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

            j3 = 1
            DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

               IF ( wate(j3,l3) .GT. 0. ) THEN
                  result(i,j,k,l,m,n) = grid(j3,l3) / wate(j3,l3)
               ELSE
                  result(i,j,k,l,m,n) = bad_flag_result
               ENDIF

               j3 = j3 + 1
 200        CONTINUE

            l3 = l3 + 1
 400     CONTINUE

         i3 = i3 + arg_incr(X_AXIS,ARG3)
 100  CONTINUE

         k3 = k3 + arg_incr(Z_AXIS,ARG3)
 300  CONTINUE

         m3 = m3 + arg_incr(E_AXIS,ARG3)
 500  CONTINUE

         n3 = n3 + arg_incr(F_AXIS,ARG3)
 600  CONTINUE

      RETURN

 999  CALL EF_BAIL_OUT(id, errtxt)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN
      END
