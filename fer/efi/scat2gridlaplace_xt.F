*
*  scat2gridlaplace_xt.F
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
* July 28 1998
*
* Returns variable interpolated onto an equally-spaced X-T grid.
* Input is scattered triples: (x, t, f(x,t)); may be functions of Y,Z,E,F.
* Output is gridded data in X,Y,Z,T,E,F.  Calls routine "zgrid_laplace".
*
* Nov 13, 2000 1) Allow modulo axes: if modulo take points from other end
*                 to use in gridding each end
*              2) Check that the scattered points are listed on the I,J,or
*                 K axis only, as they may be functions of time.
*  (12/1/2000) 3) If the destination axis is modulo, treat the scattered
*                 points as modulo too.
*
*   12/7/2000  Add error checking on gridding parameters
*    5/2001    Let F(x,y) be a function of Z and/or T
*
*  11/12/2002  Remove option 2) above: the spatial location of the scattered points
*              is a 1-D list.
*   2/ 9/2005  When we fixed the argument limit machinery this fcn
*              breaks if the output axes are limited by some context.
*              use ef_set_axis_extend to force the regridding to be
*              done on the unrestricted output grid.
*  8/26/2008   The functions call zgrid from the PPLUS code. This function has
*              some hard-wired workspace limited to 50K points. Instead allocate
*              work arrays here and pass them to zgrid_laplace.F in this directory
*              with no changes other than the work array declarations.
*
*  *kms* 4/2012 Removed extending result grid axes with ef_unspecified_int4: 
*               this appears to destroy the axes of the result grid as demonstrated
*               by the listing in bn_scat2grid_test.jnl.  Now it is up to the
*               the user to give (not truncate) the result grid axes:
*      let sgrid = scat2gridlaplace_xy(xpts, ypts, fpts, x[gx=xax5], y[gy=yax5], 5., 5)
*      load sgrid  ! <--- without this, the following uses a 1-element Y axis
*      list/x=4:5/y=1 sgrid

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


      SUBROUTINE scat2gridlaplace_xt_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 'Use Laplace weighting to grid ' //
     .                     'scattered data to an XT grid.')
      CALL ef_set_num_args(id, 7)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 8)

* XT grid is determined by arguments 4 and 5, the result's x and t axes.

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'X coordinates of scattered XT locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'T coordinates of scattered XT locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      CALL ef_set_arg_desc(id, arg,
     .        'F(X,T) data at scattered XT locations. ' //
     .        'May also be fcn of Y,Z,E,F')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, YES, NO, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'X axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, NO, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'TAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'T axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, YES, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'KAY')
      CALL ef_set_arg_desc(id, arg,
     .        'Laplace/spline mixing parameter KAY; ' //
     .        '0.0=pure Laplace; inf=pure spline')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'NRNG')
      CALL ef_set_arg_desc(id, arg,
     .        'Effect parameter NRNG; grid vals ' //
     .        'undef if > NRNG grid steps from pts')
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
      SUBROUTINE scat2gridlaplace_xt_work_size(id)

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
      INTEGER nxout, ntout, nx2, nt2
      INTEGER nxin, nyin, nzin, ntin, nein, nfin, nin

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxin = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      nyin = 1 + arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)
      nzin = 1 + arg_hi_ss(Z_AXIS,ARG1) - arg_lo_ss(Z_AXIS,ARG1)
      ntin = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)
      nein = 1 + arg_hi_ss(E_AXIS,ARG1) - arg_lo_ss(E_AXIS,ARG1)
      nfin = 1 + arg_hi_ss(F_AXIS,ARG1) - arg_lo_ss(F_AXIS,ARG1)
      nin = MAX(nxin,nyin,nzin,ntin,nein,nfin)
*  If modulo axis, the scatter arrays are extended for values
*  within nrng of the cut-point.  Worst case: everything gets
*  duplicated at both ends; thus three times the points for
*  each modulo axis.
      nin = 9 * nin

      nxout = 1 + arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4)
      ntout = 1 + arg_hi_ss(T_AXIS,ARG5) - arg_lo_ss(T_AXIS,ARG5)

      nx2 = nxout * 2
      nt2 = ntout * 2

* xax  output x axis
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                              nx2, 1, 1, 1, 1, 1)

* tax  output t axis
      CALL ef_set_work_array_dims_6d(id, 2,
     .                                1, 1, 1, 1, 1, 1,
     .                              nt2, 1, 1, 1, 1, 1)

* zgridded  work array - gridded data.
      CALL ef_set_work_array_dims_6d(id, 3,
     .                                1,     1, 1, 1, 1, 1,
     .                            nxout, ntout, 1, 1, 1, 1)

* xscat - scattered points with gaps removed and extended for modulo
      CALL ef_set_work_array_dims_6d(id, 4,
     .                                1, 1, 1, 1, 1, 1,
     .                              nin, 1, 1, 1, 1, 1)

* tscat - scattered points with gaps removed and extended for modulo
      CALL ef_set_work_array_dims_6d(id, 5,
     .                                1, 1, 1, 1, 1, 1,
     .                              nin, 1, 1, 1, 1, 1)

* fscat - scattered points with gaps removed and extended for modulo
      CALL ef_set_work_array_dims_6d(id, 6,
     .                                1, 1, 1, 1, 1, 1,
     .                              nin, 1, 1, 1, 1, 1)

* zgrid_laplace work array zpij
      CALL ef_set_work_array_dims_6d(id, 7,
     .                                1, 1, 1, 1, 1, 1,
     .                              nin, 1, 1, 1, 1, 1)

* zgrid_laplace work array knxt
      CALL ef_set_work_array_dims_6d(id, 8,
     .                                1, 1, 1, 1, 1, 1,
     .                              nin, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE scat2gridlaplace_xt_compute(id, arg_1, arg_2, arg_3,
     .                            arg_4, arg_5, arg_6, arg_7, result,
     .                            xax, tax, zgridded, xscat, tscat,
     .                            fscat, zpij, knxt)

*  arg_1  xpts \
*  arg_2  tpts  > scattered x,t,f(x,t) triples to be gridded. F can be fcn of y,z,e,f
*  arg_3  Fpts /
*  arg_4  xaxis of new grid
*  arg_5  taxis of new grid
*  arg_6  interpolation parameter   kay
*  arg_7  interpolation parameter   nrng

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

      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* Ignore the unused final dimensions in the work arrays since
* Fortran is column major
      REAL*8 xax(wrk1lox:wrk1hix/2)
      REAL*8 tax(wrk2lox:wrk2hix/2)
      REAL zgridded(wrk3lox:wrk3hix, wrk3loy:wrk3hiy)
      REAL xscat(wrk4lox:wrk4hix)
      REAL tscat(wrk5lox:wrk5hix)
      REAL fscat(wrk6lox:wrk6hix)
      REAL zpij(wrk7lox:wrk7hix)
      REAL knxt(wrk8lox:wrk8hix)

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

      REAL zgridbad
      PARAMETER (zgridbad = 1.e+35)  ! missing-data flag used by zgrid_laplace

      INTEGER i, j, k, l, m, n
      INTEGER idx, lo_ss, hi_ss
      INTEGER nxpts, ntpts, npts, nscat
      INTEGER nx, nt
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER xlo, xhi, tlo, thi
      INTEGER nrng
      REAL x1, t1, dx, dt
      REAL kay
      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo, regular)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), mdulo(6), regular(6)
      LOGICAL modulox, modulot
      REAL*8 modxlen, modtlen

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
      IF ( .NOT. regular(T_AXIS) ) THEN
         errtxt = 'Output axes must be REGULARLY spaced'
         GOTO 999
      ENDIF
      modulot = mdulo(T_AXIS)
      IF ( modulot ) THEN
         CALL ef_get_axis_modulo_len(id, ARG5, T_AXIS, modtlen)
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

      IF ( nxpts .NE. ntpts ) THEN
         WRITE (errtxt,90) ' XPTS,', ' TPTS ', nxpts, ntpts
         GOTO 999
      ENDIF

* Verify the number of values given in ARG3 makes sense
      xlo = arg_lo_ss(X_AXIS,ARG3)
      xhi = arg_hi_ss(X_AXIS,ARG3)
      tlo = arg_lo_ss(T_AXIS,ARG3)
      thi = arg_hi_ss(T_AXIS,ARG3)

      IF ( (xhi .GT. xlo) .AND. (thi .GT. tlo) ) THEN
         errtxt = 'F() data values defined on both X and T axis'
         GOTO 999
      ELSE IF ( xhi .GT. xlo ) THEN
         npts = xhi - xlo + 1
         IF ( nxpts .NE. npts ) THEN
            WRITE (errtxt,90) ' XPTS,', ' F(X) ', nxpts, npts
            GOTO 999
         ENDIF
      ELSE IF ( thi .GT. tlo ) THEN
         npts = thi - tlo + 1
         IF ( nxpts .NE. npts ) THEN
            WRITE (errtxt,90) ' XPTS,', ' F(T) ', nxpts, npts
            GOTO 999
         ENDIF
      ELSE IF ( (xlo .NE. ef_unspecified_int4) .OR.
     .          (tlo .NE. ef_unspecified_int4) ) THEN
         npts = 1
         IF ( nxpts .NE. npts ) THEN
            WRITE (errtxt,90) ' XPTS,', ' F()  ', nxpts, npts
            GOTO 999
         ENDIF
      ELSE
         errtxt = 'No data in scattered x, t points.  ' //
     .            'F() data values must be defined on X or T axis.'
         GOTO 999
      ENDIF
  90  FORMAT('Input scattered', 2A6, 'have different # of points', 2I8)

*  Compute number of points in output axes.

      lo_ss = arg_lo_ss(X_AXIS,ARG4)
      hi_ss = arg_hi_ss(X_AXIS,ARG4)
*  Check that xax is an X axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fourth argument must be an X axis'
         GO TO 999
      ENDIF
      nx = hi_ss - lo_ss + 1

      lo_ss = arg_lo_ss(T_AXIS,ARG5)
      hi_ss = arg_hi_ss(T_AXIS,ARG5)
*  Check that tax a T axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fifth argument must be a T axis'
         GO TO 999
      ENDIF
      nt = hi_ss - lo_ss + 1

C  Get coordinates of output axes.

      CALL ef_get_coordinates(id, ARG4, X_AXIS, arg_lo_ss(X_AXIS, ARG4),
     .                        arg_hi_ss(X_AXIS, ARG4), xax)

      CALL ef_get_coordinates(id, ARG5, T_AXIS, arg_lo_ss(T_AXIS, ARG5),
     .                        arg_hi_ss(T_AXIS, ARG5), tax)

*  Set start and delta for output axes.

      x1 = xax(1)
      t1 = tax(1)

      IF ( nx .GT. 1 ) THEN
         dx = xax(2) - xax(1)
      ELSE
         errtxt = 'More than one output X coordinate required'
         GOTO 999
      ENDIF
      IF ( nt .GT. 1 ) THEN
         dt = tax(2) - tax(1)
      ELSE
         errtxt = 'More than one output T coordinate required'
         GOTO 999
      ENDIF

*  Get interpolation parameters.

      kay = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6),
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6),
     .            arg_lo_ss(E_AXIS,ARG6), arg_lo_ss(F_AXIS,ARG6))
      IF ( kay .LT. 0. ) THEN
         errtxt = 'Gridding parameter KAY must be nonnegative'
         GOTO 999
      ENDIF

      nrng = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7),
     .             arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7),
     .             arg_lo_ss(E_AXIS,ARG7), arg_lo_ss(F_AXIS,ARG7))
      IF ( nrng .LE. 0 ) THEN
         errtxt = 'Gridding parameter NRNG must be a positive integer'
         GOTO 999
      ENDIF

*  Compute result at each Y, Z, E, F

      n3 = arg_lo_ss(F_AXIS,ARG3)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m3 = arg_lo_ss(E_AXIS,ARG3)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      k3 = arg_lo_ss(Z_AXIS,ARG3)
      DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

      j3 = arg_lo_ss(Y_AXIS,ARG3)
      DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

*  If there is a missing value routine zgrid does not grid the data in
*  a zone around that point, even if there are other nearby good data
*  points.  Remove missing/bad data from the input (x,t,f) points.
*  Assigns xscat, tscat, fscat, and nscat.

         CALL nobadxt(arg_1, arg_2, arg_3,
     .                bad_flag(ARG1), bad_flag(ARG2), bad_flag(ARG3),
     .                npts, xlo, xhi, j3, k3, tlo, thi, m3, n3,
     .                xscat, tscat, fscat, nscat)

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered points.
         IF ( modulox ) THEN
            CALL modscat(xax(1), xax(nx), modxlen, nscat, xscat)
         ENDIF
         IF ( modulot ) THEN
            CALL modscat(tax(1), tax(nt), modtlen, nscat, tscat)
         ENDIF

*  If an output axis is modulo, mirror the scattered points within NRNG
*  of each end to the other end of the region.  Extends xscat, tscat,
*  fscat, and updates nscat to the new count.
         IF ( modulox ) THEN
            CALL copyscat(xax, nx, nrng, nscat, xscat, tscat, fscat)
         ENDIF
         IF ( modulot ) THEN
            CALL copyscat(tax, nt, nrng, nscat, tscat, xscat, fscat)
         ENDIF

*  Initialize zgridded to all zeros

         DO l3 = 1, nt
            DO i3 = 1, nx
               zgridded(i3,l3) = 0.0
            ENDDO
         ENDDO

*  Grid the data.

         CALL zgrid_laplace(zgridded, nx, nt, nx, nt, x1, t1,
     .      dx, dt, xscat, tscat, fscat, zpij, knxt, nscat, kay, nrng)

*  Put zgridded into result variable.  Use "bad_flag_result" rather than
*  "zgridbad" to mark bad data.

         l3 = 1
         DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

            i3 = 1
            DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

               IF ( zgridded(i3,l3) .EQ. zgridbad ) THEN
                  result(i,j,k,l,m,n) = bad_flag_result
               ELSE
                  result(i,j,k,l,m,n) = zgridded(i3,l3)
               ENDIF

               i3 = i3 + 1
  100       CONTINUE

            l3 = l3 + 1
  400    CONTINUE

         j3 = j3 + arg_incr(Y_AXIS,ARG3)
  200 CONTINUE

         k3 = k3 + arg_incr(Z_AXIS,ARG3)
  300 CONTINUE

         m3 = m3 + arg_incr(E_AXIS,ARG3)
  500 CONTINUE

         n3 = n3 + arg_incr(F_AXIS,ARG3)
  600 CONTINUE

      RETURN

  999 CALL EF_BAIL_OUT(id, errtxt)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN
      END


      SUBROUTINE nobadxt(xin, tin, fin, badx, badt, badf,
     .                   npts, xlo, xhi, j3, k3, tlo, thi, m3, n3,
     .                   xscat, tscat, fscat, nscat)

* ACM 11/00 change so not exact comparison: wasnt catching all bad values.
* ACM  5/01 properly deal with f-scattered points which may be fcns of y and z

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      LOGICAL TM_FPEQ

*  Input arguments
      REAL xin(*), tin(*)
      REAL fin(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz,
     .         mem3lot:mem3hit, mem3loe:mem3hie, mem3lof:mem3hif)
      REAL badx, badt, badf
      INTEGER npts, xlo, xhi, j3, k3, tlo, thi, m3, n3

*  Output arguments
      REAL xscat(*), tscat(*), fscat(*)
      INTEGER nscat

*  Local variables
      INTEGER idx, i3, l3

      nscat = 0

      IF ( xhi .GT. xlo ) THEN

         DO 100 idx = 1, npts
            IF ( TM_FPEQ(xin(idx), badx) ) GOTO 100
            IF ( TM_FPEQ(tin(idx), badt) ) GOTO 100
            i3 = xlo + idx - 1
            l3 = tlo
            IF ( TM_FPEQ(fin(i3,j3,k3,l3,m3,n3), badf) ) GOTO 100
            nscat = nscat + 1
            xscat(nscat) = xin(idx)
            tscat(nscat) = tin(idx)
            fscat(nscat) = fin(i3,j3,k3,l3,m3,n3)
  100    CONTINUE

      ELSE

         DO 200 idx = 1, nscat
            IF ( TM_FPEQ(xin(idx), badx) ) GOTO 200
            IF ( TM_FPEQ(tin(idx), badt) ) GOTO 200
            i3 = xlo
            l3 = tlo + idx - 1
            IF ( TM_FPEQ(fin(i3,j3,k3,l3,m3,n3), badf) ) GOTO 200
            nscat = nscat + 1
            xscat(nscat) = xin(idx)
            tscat(nscat) = tin(idx)
            fscat(nscat) = fin(i3,j3,k3,l3,m3,n3)
  200    CONTINUE

      ENDIF

      RETURN
      END


C**
C**  Comments from routine ZGRID used in the PLOT+ graphics package; 
C**  the code is part of Ferret which calls PLOT+
C**
C**	subroutine zgrid(z,nxsize,nysize,nx,ny,x1,y1,dx,dy, 
C**     1	xp,yp,zp,n,cay,nrng) 
C** 
C**    @(#)zgrid.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c     sets up square grid for contouring , given arbitrarily placed 
c     data points. laplace interpolation is used. 
c     the method used here was lifted directly from notes left by 
c     mr ian crain formerly with the comp.science div. 
c     info on relaxation soln of laplace eqn supplied by dr t murty. 
c     fortran ii   oceanography/emr   dec/68   jdt 
c 
c     z = 2-d array of hgts to be set up. points outside region to be 
c     contoured should be initialized to 10**35 . the rest should be 0.0 
c     nx,ny = max subscripts of z in x and y directions . 
c     x1,y1 = coordinates of z(1,1) 
c     dx,dy = x and y increments . 
c     xp,yp,zp = arrays giving position and hgt of each data point. 
c     n = size of arrays xp,yp and zp . 
c 
c     modification feb/69   to get smoother results a portion of the 
c     beam eqn  was added to the laplace eqn giving 
c     delta2x(z)+delta2y(z) - k(delta4x(z)+delta4y(z)) = 0 . 
c     k=0 gives pure laplace solution.  k=inf. gives pure spline solution. 
c     cayin = k = amount of spline eqn (between 0 and inf.) 
c     nrng...grid points more than nrng grid spaces from the nearest 
c            data point are set to undefined. 
c 
c     modification dec23/69   data pts no longer moved to grid pts. 
c 
c     modification may 5 79  common blocks work1 and work2 must 
c     be dimension at least n points long by the user.  common 
c     block work3 must be dimensioned at least ny points long. 
c 
c	modification june 17,1985 - handles data values of 1e35. if at
c	least one data value near a grid point is equal to 1e35, the z
c	array is initialized to 1e35 at that grid point
c	- by g.r. halliwell
c
