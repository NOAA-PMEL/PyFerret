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
*               and cutoff.
*
* Nov 13, 2000 1) Allow modulo axes: if modulo take points from other end
*                 to use in gridding each end
*              2) Check that the scattered points are listed on the I,J,or
*                 K axis only, as they may be functions of time.
*  (12/1/2000) 3) If the destination axis is modulo, treat the scattered
*                 points as modulo too.
*
*   12/7/2000  Add error checking on gridding parameters
*
*    5/2001    Let F(x,z) be a function of Y and/or T
*
*  11/12/2002  Remove option 2) above: the spatial location of the scattered points
*              is a 1-D list.

* Returns variable interpolated onto an equally-spaced X-Y grid.
* Input is scattered triples: (x, z, f(x,z)); may be functions of y and/or time.
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

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 'Use Gaussian weighting to grid ' //
     .                     'scattered data to an XZ grid.')
      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

* XZ grid is determined by arguments 4 and 5, the result's x and z axes.

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'X coordinates of scattered XZ locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'ZPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Z coordinates of scattered XZ locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      CALL ef_set_arg_desc(id, arg,
     .        'F(X,Z) data at scattered XZ locations. ' //
     .        'May also be fcn of Y,T,E,F')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, NO, YES, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'X axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, NO, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'ZAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Z axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, YES, NO, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'XSCALE')
      CALL ef_set_arg_desc(id, arg,
     .  'Mapping scale in X direction, in data units (e.g. km or lon)')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'ZSCALE')
      CALL ef_set_arg_desc(id, arg,
     .    'Mapping scale in Z direction, in data units (e.g. m or km)')
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
      SUBROUTINE scat2gridgauss_xz_work_size(id)

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
      INTEGER nxout, nzout, nx2, nz2

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4)
      nzout = 1 + arg_hi_ss(Z_AXIS,ARG5) - arg_lo_ss(Z_AXIS,ARG5)

      nx2 = nxout * 2
      nz2 = nzout * 2

* xax  output x axis
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                              nx2, 1, 1, 1, 1, 1)

* zax  output z axis
      CALL ef_set_work_array_dims_6d(id, 2,
     .                                1, 1, 1, 1, 1, 1,
     .                              nz2, 1, 1, 1, 1, 1)

* grid  work array - gridded data.
      CALL ef_set_work_array_dims_6d(id, 3,
     .                                1,     1, 1, 1, 1, 1,
     .                            nxout, nzout, 1, 1, 1, 1)

* wate - weights.
      CALL ef_set_work_array_dims_6d(id, 4,
     .                                1,     1, 1, 1, 1, 1,
     .                            nxout, nzout, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE scat2gridgauss_xz_compute(id, arg_1, arg_2, arg_3,
     .                           arg_4, arg_5, arg_6, arg_7, arg_8,
     .                           arg_9, result, xax, zax, grid, wate)

*  arg_1  xpts \
*  arg_2  zpts  > Scattered x,z,f(x,z) triples to be gridded.
*  arg_3  fpts /    Can be fcn of y,t,e,f
*  arg_4  X axis of new grid
*  arg_5  Z axis of new grid
*  arg_6  interpolation parameter   xscale
*  arg_7  interpolation parameter   zscale
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
      REAL*8 xax(wrk1lox:wrk1hix/2)
      REAL*8 zax(wrk2lox:wrk2hix/2)
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
      INTEGER nxpts, nzpts, nscat
      INTEGER nx, nz, nt
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER xlo, xhi, zlo, zhi
      REAL x1, z1, t1, xf, zf, tf
      REAL xx, zz, tt
      REAL xsc, zsc, tsc
      REAL cutoff
      REAL val
      REAL dx, dz, xcut, zcut, xxbeg, xxend, zzbeg, zzend
      INTEGER iwflag
      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), mdulo(6), regular(6)
      LOGICAL modulox, moduloz
      REAL*8 modxlen, modzlen

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
      IF ( .NOT. regular(Z_AXIS) ) THEN
         errtxt = 'Output axes must be REGULARLY spaced'
         GOTO 999
      ENDIF
      moduloz = mdulo(Z_AXIS)
      IF ( moduloz ) THEN
         CALL ef_get_axis_modulo_len(id, ARG5, Z_AXIS, modzlen)
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

      nzpts = 0
      DO 30 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG2)
         hi_ss = arg_hi_ss(idx,ARG2)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( nzpts .NE. 0 ) THEN
               nzpts = nzpts * (hi_ss - lo_ss + 1)
            ELSE
               nzpts = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  30  CONTINUE

      IF ( nxpts .NE. nzpts ) THEN
         WRITE (errtxt,90) ' XPTS,', ' ZPTS ', nxpts, nzpts
         GOTO 999
      ENDIF

* Verify the number of values given in ARG3 makes sense
      xlo = arg_lo_ss(X_AXIS,ARG3)
      xhi = arg_hi_ss(X_AXIS,ARG3)
      zlo = arg_lo_ss(Z_AXIS,ARG3)
      zhi = arg_hi_ss(Z_AXIS,ARG3)

      IF ( (xhi .GT. xlo) .AND. (zhi .GT. zlo) ) THEN
         errtxt = 'F() data values defined on both X and Z axis'
         GOTO 999
      ELSE IF ( xhi .GT. xlo ) THEN
         nscat = xhi - xlo + 1
         IF ( nxpts .NE. nscat ) THEN
            WRITE (errtxt,90) ' XPTS,', ' F(X) ', nxpts, nscat
            GOTO 999
         ENDIF
      ELSE IF ( zhi .GT. zlo ) THEN
         nscat = zhi - zlo + 1
         IF ( nxpts .NE. nscat ) THEN
            WRITE (errtxt,90) ' XPTS,', ' F(Z) ', nxpts, nscat
            GOTO 999
         ENDIF
      ELSE IF ( (xlo .NE. ef_unspecified_int4) .OR.
     .          (zlo .NE. ef_unspecified_int4) ) THEN
         nscat = 1
         IF ( nxpts .NE. nscat ) THEN
            WRITE (errtxt,90) ' XPTS,', ' F()  ', nxpts, nscat
            GOTO 999
         ENDIF
      ELSE
         errtxt = 'No data in scattered x, z points.  ' //
     .            'F() data values must be defined on X or Z axis.'
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

      lo_ss = arg_lo_ss(Z_AXIS,ARG5)
      hi_ss = arg_hi_ss(Z_AXIS,ARG5)
*  Check that zax a Z axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fifth argument must be a Z axis'
         GO TO 999
      ENDIF
      nz = hi_ss - lo_ss + 1

C  Get coordinates of output axes.

      CALL ef_get_coordinates(id, ARG4, X_AXIS, arg_lo_ss(X_AXIS, ARG4),
     .                        arg_hi_ss(X_AXIS, ARG4), xax)

      CALL ef_get_coordinates(id, ARG5, Z_AXIS, arg_lo_ss(Z_AXIS, ARG5),
     .                        arg_hi_ss(Z_AXIS, ARG5), zax)

*  Set start and end for output axes.

      x1 = xax(1)
      z1 = zax(1)

      xf = xax(nx)
      zf = zax(nz)

*  Time parameters for subroutine gausswt.  Calling with 1 time.

      nt = 1
      tt = 1.
      t1 = 1.
      tf = 1.
      tsc = 1.

*  iwflag=1 for time wrapping; 0 for no wrapping

      iwflag = 0

*  Get interpolation parameters: mapping scales (data units)

      xsc = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6),
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6),
     .            arg_lo_ss(E_AXIS,ARG6), arg_lo_ss(F_AXIS,ARG6))
      IF ( xsc .LE. 0. ) THEN
         errtxt = 'X mapping scale parameter must be positive'
         GOTO 999
      ENDIF

      zsc = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7),
     .            arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7),
     .            arg_lo_ss(E_AXIS,ARG7), arg_lo_ss(F_AXIS,ARG7))
      IF ( tsc .LE. 0. ) THEN
         errtxt = 'Z mapping scale parameter must be positive'
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

*  Compute result at each Y, T, E, F

      n3 = arg_lo_ss(F_AXIS,ARG3)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m3 = arg_lo_ss(E_AXIS,ARG3)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      l3 = arg_lo_ss(T_AXIS,ARG3)
      DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

      j3 = arg_lo_ss(Y_AXIS,ARG3)
      DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

*  Initialize sums of values and weights.

         DO k3 = 1, nz
            DO i3 = 1, nx
               grid(i3,k3) = 0.
            ENDDO
         ENDDO

         DO k3 = 1, nz
            DO i3 = 1, nx
               wate(i3,k3) = 0.
            ENDDO
         ENDDO

*  Loop over x and z, compute the weighted sums for gaussian-weighted mapping
*  onto the grid.  Args 1-2 may be on the X,Y,Z,T,E or F axis of ARG1 and
*  ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

         DO 700 idx = 1, nscat

            CALL pickout2(arg_1, arg_2, idx, xx, zz)
            IF ( xhi .GT. xlo ) THEN
               i3 = xlo + idx - 1
               k3 = zlo
            ELSE
               i3 = xlo
               k3 = zlo + idx - 1
            ENDIF
            val = arg_3(i3,j3,k3,l3,m3,n3)

            IF (  xx .EQ. bad_flag(ARG1) .OR.
     .            zz .EQ. bad_flag(ARG2) .OR.
     .           val .EQ. bad_flag(ARG3) ) GOTO 700

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
            IF ( modulox ) THEN
               CALL modscat(xax(1), xax(nx), modxlen, 1, xx)
            ENDIF
            IF ( moduloz ) THEN
               CALL modscat(zax(1), zax(nz), modzlen, 1, zz)
            ENDIF

            CALL gausswt2(xx, zz, tt, val, grid, wate, nx,
     .                    nz, nt, x1, z1, t1, xf, zf, tf, xsc,
     .                    zsc, tsc, cutoff, iwflag, nx, nz)

C ACM modulo 11/9/00  Put points within cutoff of the end just beyond the
C                     other end, and use in the gridding computation.

            IF ( modulox ) THEN
               dx = xf - x1
               IF (nx .GT. 1) dx = (xf-x1)/real(nx-1)  ! gridbox size in data units
               xcut = cutoff*xsc/dx                    ! cutoff scaled to grid units

               IF ( (xx-x1 .GE. 0.) .AND. (xx-x1 .LT. xcut) ) THEN
                  xxend = xf + (xx-x1)
                  CALL gausswt2(xxend, zz, tt, val, grid, wate, nx,
     .                          nz, nt, x1, z1, t1, xf, zf, tf, xsc,
     .                          zsc, tsc, cutoff, iwflag, nx, nz)
               ENDIF

               IF ( (xf-xx .GE. 0.) .AND. (xf-xx .LT. xcut) ) THEN
                  xxbeg = x1 - (xf-xx)
                  CALL gausswt2(xxbeg, zz, tt, val, grid, wate, nx,
     .                          nz, nt, x1, z1, t1, xf, zf, tf, xsc,
     .                          zsc, tsc, cutoff, iwflag, nx, nz)
               ENDIF
            ENDIF

            IF ( moduloz ) THEN
               dz = zf - z1
               IF (nz .GT. 1) dz = (zf-z1)/real(nz-1)  ! gridbox size in data units
               zcut = cutoff*zsc/dz                    ! cutoff scaled to grid units

               IF ( (zz-z1 .GE. 0.) .AND. (zz-z1 .LT. zcut) ) THEN
                  zzend = zf + (zz-z1)
                  CALL gausswt2(xx, zzend, tt, val, grid, wate, nx,
     .                          nz, nt, x1, z1, t1, xf, zf, tf, xsc,
     .                          zsc, tsc, cutoff, iwflag, nx, nz)
               ENDIF

               IF ( (zf-zz .LT. 0.) .AND. (zf-zz .LT. zcut) ) THEN
                  zzbeg = z1 - (zf-zz)
                  CALL gausswt2(xx, zzbeg, tt, val, grid, wate, nx,
     .                          nz, nt, x1, z1, t1, xf, zf, tf, xsc,
     .                          zsc, tsc, cutoff, iwflag, nx, nz)
               ENDIF
            ENDIF

 700     CONTINUE

*  Put gridded fcn into result variable, dividing by summed weights. (as in
*  gaussfin, but indices needn't start at 1)

         k3 = 1
         DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            i3 = 1
            DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

               IF ( wate(i3,k3) .GT. 0. ) THEN
                  result(i,j,k,l,m,n) = grid(i3,k3) / wate(i3,k3)
               ELSE
                  result(i,j,k,l,m,n) = bad_flag_result
               ENDIF

               i3 = i3 + 1
 100        CONTINUE

            k3 = k3 + 1
 300     CONTINUE

         j3 = j3 + arg_incr(Y_AXIS,ARG3)
 200  CONTINUE

         l3 = l3 + arg_incr(T_AXIS,ARG3)
 400  CONTINUE

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
