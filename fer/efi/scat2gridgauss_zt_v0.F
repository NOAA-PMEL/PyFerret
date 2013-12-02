*
*  scat2gridgauss_zt_v0
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

* Returns variable interpolated onto an equally-spaced Z-T grid.
* Input is scattered triples: (z, t, variable) , f may be function of x, y, e, f
* Output is gridded data in z, t.  Calls routine "gausswt2_v0".
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


      SUBROUTINE  scat2gridgauss_zt_v0_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 'Use Gaussian v0 weighting to grid ' //
     .                     'scattered data to a ZT grid.')
      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

* ZT grid is determined by arguments 4 and 5, the result's z and t axes.

      arg = 1
      CALL ef_set_arg_name(id, arg, 'ZPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Z coordinates of scattered ZT locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'T coordinates of scattered ZT locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      CALL ef_set_arg_desc(id, arg,
     .        'variable at scattered ZT locations. ' //
     .        'May also be fcn of X,Y,E,F')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, NO, NO, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'ZAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Z axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, YES, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'TAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'T axis coordinates of a regular output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, YES, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'ZSCALE')
      CALL ef_set_arg_desc(id, arg,
     .    'Mapping scale in Z direction, in data units (e.g. m or km)')
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
      SUBROUTINE scat2gridgauss_zt_v0_work_size(id)

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
      INTEGER nzout, ntout, nz2, nt2

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nzout = 1 + arg_hi_ss(Z_AXIS,ARG4) - arg_lo_ss(Z_AXIS,ARG4)
      ntout = 1 + arg_hi_ss(T_AXIS,ARG5) - arg_lo_ss(T_AXIS,ARG5)

      nz2 = nzout * 2
      nt2 = ntout * 2

* zax  output z axis
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                              nz2, 1, 1, 1, 1, 1)

* tax  output t axis
      CALL ef_set_work_array_dims_6d(id, 2,
     .                                1, 1, 1, 1, 1, 1,
     .                              nt2, 1, 1, 1, 1, 1)

* grid  work array - gridded data.
      CALL ef_set_work_array_dims_6d(id, 3,
     .                                1,     1, 1, 1, 1, 1,
     .                            nzout, ntout, 1, 1, 1, 1)

* wate - weights.
      CALL ef_set_work_array_dims_6d(id, 4,
     .                                1,     1, 1, 1, 1, 1,
     .                            nzout, ntout, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE scat2gridgauss_zt_v0_compute(id, arg_1, arg_2, arg_3,
     .                           arg_4, arg_5, arg_6, arg_7, arg_8,
     .                           arg_9, result, zax, tax, grid, wate)

*  arg_1  ypts \
*  arg_2  tpts  > Scattered z,t,variable,  triples to be gridded.
*  arg_3  variable, /    variable, can be fcn of x,y,e,f
*  arg_4  z axis of output grid
*  arg_5  t axis of output grid
*  arg_6  interpolation parameter   zscale
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
      REAL*8 zax(wrk1lox:wrk1hix/2)
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
      INTEGER nzpts, ntpts, nscat
      INTEGER nx, nz, nt
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER zlo, zhi, tlo, thi
      REAL x1, z1, t1, xf, zf, tf
      REAL xx, zz, tt
      REAL xsc, zsc, tsc
      REAL cutoff
      REAL val
      REAL dz, dt, zcut, tcut, zzbeg, zzend, ttbeg, ttend
      INTEGER iwflag
      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), mdulo(6), regular(6)
      LOGICAL moduloz, modulot
      REAL*8 modzlen, modtlen

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Check to see if output axes are regular, and get modulo characteristics
      CALL ef_get_axis_info_6d(id, ARG4, ax_name, ax_units,
     .                         backward, mdulo, regular)
      IF ( .NOT. regular(Z_AXIS) ) THEN
         errtxt = 'Output axes must be REGULARLY spaced'
         GOTO 999
      ENDIF
      moduloz = mdulo(Z_AXIS)
      IF ( moduloz ) THEN
         CALL ef_get_axis_modulo_len(id, ARG4, Z_AXIS, modzlen)
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

      nzpts = 0
      DO 30 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG1)
         hi_ss = arg_hi_ss(idx,ARG1)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( nzpts .NE. 0 ) THEN
               nzpts = nzpts * (hi_ss - lo_ss + 1)
            ELSE
               nzpts = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  30  CONTINUE

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

      IF ( nzpts .NE. ntpts ) THEN
         WRITE (errtxt,90) ' ZPTS,', ' TPTS ', nzpts, ntpts
         GOTO 999
      ENDIF

* Verify the number of values given in ARG3 makes sense
      zlo = arg_lo_ss(Z_AXIS,ARG3)
      zhi = arg_hi_ss(Z_AXIS,ARG3)
      tlo = arg_lo_ss(T_AXIS,ARG3)
      thi = arg_hi_ss(T_AXIS,ARG3)

      IF ( (zhi .GT. zlo) .AND. (thi .GT. tlo) ) THEN
         errtxt = 'F() data values defined on both Z and T axis'
         GOTO 999
      ELSE IF ( zhi .GT. zlo ) THEN
         nscat = zhi - zlo + 1
         IF ( nzpts .NE. nscat ) THEN
            WRITE (errtxt,90) ' ZPTS,', ' F(Z) ', nzpts, nscat
            GOTO 999
         ENDIF
      ELSE IF ( thi .GT. tlo ) THEN
         nscat = thi - tlo + 1
         IF ( nzpts .NE. nscat ) THEN
            WRITE (errtxt,90) ' ZPTS,', ' F(T) ', nzpts, nscat
            GOTO 999
         ENDIF
      ELSE IF ( (zlo .NE. ef_unspecified_int4) .OR.
     .          (tlo .NE. ef_unspecified_int4) ) THEN
         nscat = 1
         IF ( nzpts .NE. nscat ) THEN
            WRITE (errtxt,90) ' ZPTS,', ' F()  ', nzpts, nscat
            GOTO 999
         ENDIF
      ELSE
         errtxt = 'No data in scattered z, t points.  ' //
     .            'F() data values must be defined on Z or T axis.'
         GOTO 999
      ENDIF
  90  FORMAT('Input scattered', 2A6, 'have different # of points', 2I8)

*  Compute number of points in output axes.

      lo_ss = arg_lo_ss(Z_AXIS,ARG4)
      hi_ss = arg_hi_ss(Z_AXIS,ARG4)
*  Check that zax is a Z axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fourth argument must be a Z axis'
         GO TO 999
      ENDIF
      nz = hi_ss - lo_ss + 1

      lo_ss = arg_lo_ss(T_AXIS,ARG5)
      hi_ss = arg_hi_ss(T_AXIS,ARG5)
*  Check that tax a T axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fifth argument must be a T axis'
         GO TO 999
      ENDIF
      nt = hi_ss - lo_ss + 1

C  Get coordinates of output axes.

      CALL ef_get_coordinates(id, ARG4, Z_AXIS, arg_lo_ss(Z_AXIS, ARG4),
     .                        arg_hi_ss(Z_AXIS, ARG4), zax)

      CALL ef_get_coordinates(id, ARG5, T_AXIS, arg_lo_ss(T_AXIS, ARG5),
     .                        arg_hi_ss(T_AXIS, ARG5), tax)

*  Set start and end for output axes.

      z1 = zax(1)
      t1 = tax(1)

      zf = zax(nz)
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

      zsc = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6),
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6),
     .            arg_lo_ss(E_AXIS,ARG6), arg_lo_ss(F_AXIS,ARG6))
      IF ( zsc .LE. 0. ) THEN
         errtxt = 'Z mapping scale parameter must be positive'
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

*  Compute result at each X, Y, E, F

      n3 = arg_lo_ss(F_AXIS,ARG3)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m3 = arg_lo_ss(E_AXIS,ARG3)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      j3 = arg_lo_ss(Y_AXIS,ARG3)
      DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

      i3 = arg_lo_ss(X_AXIS,ARG3)
      DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

*  Initialize sums of values and weights.

         DO l3 = 1, nt
            DO k3 = 1, nz
               grid(k3,l3) = 0.
            ENDDO
         ENDDO

         DO l3 = 1, nt
            DO k3 = 1, nz
               wate(k3,l3) = 0.
            ENDDO
         ENDDO

*  Loop over z and t, compute the weighted sums for gaussian-weighted mapping
*  onto the grid.  Args 1-2 may be on the X,Y,Z,T,E, or F axis of ARG1 and
*  ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

         DO 700 idx = 1, nscat

            CALL pickout2(arg_1, arg_2, idx, zz, tt)
            IF ( zhi .GT. zlo ) THEN
               k3 = zlo + idx - 1
               l3 = tlo
            ELSE
               k3 = zlo
               l3 = tlo + idx - 1
            ENDIF
            val = arg_3(i3,j3,k3,l3,m3,n3)

            IF (  zz .EQ. bad_flag(ARG1) .OR.
     .            tt .EQ. bad_flag(ARG2) .OR.
     .           val .EQ. bad_flag(ARG3) ) GOTO 700

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
            IF ( moduloz ) THEN
               CALL modscat(zax(1), zax(nz), modzlen, 1, zz)
            ENDIF
            IF ( modulot ) THEN
               CALL modscat(tax(1), tax(nt), modtlen, 1, tt)
            ENDIF

            CALL gausswt2_v0(zz, tt, xx, val, grid, wate, nz,
     .                    nt, nx, z1, t1, x1, zf, tf, xf, zsc,
     .                    tsc, xsc, cutoff, iwflag, nz, nt)

C ACM modulo 11/9/00  Put points within cutoff of the end just beyond the
C                     other end, and use in the gridding computation.

            IF ( moduloz ) THEN
               dz = zf - z1
               IF (nz .GT. 1) dz = (zf-z1)/real(nz-1)  ! gridbox size in data units
               zcut = cutoff*zsc/dz                    ! cutoff scaled to grid units

               IF ( (zz-z1 .GE. 0.) .AND. (zz-z1 .LT. zcut) ) THEN
                  zzend = zf + (zz-z1)
                  CALL gausswt2_v0(zzend, tt, xx, val, grid, wate, nz,
     .                          nt, nx, z1, t1, x1, zf, tf, xf, zsc,
     .                          tsc, xsc, cutoff, iwflag, nz, nt)
               ENDIF

               IF ( (zf-zz .GE. 0.) .AND. (zf-zz .LT. zcut) ) THEN
                  zzbeg = z1 - (zf-zz)
                  CALL gausswt2_v0(zzbeg, tt, xx, val, grid, wate, nz,
     .                          nt, nx, z1, t1, x1, zf, tf, xf, zsc,
     .                          tsc, xsc, cutoff, iwflag, nz, nt)
               ENDIF
            ENDIF

            IF ( modulot ) THEN
               dt = tf - t1
               IF (nt .GT. 1) dt = (tf-t1)/real(nt-1)  ! gridbox size in data units
               tcut = cutoff*tsc/dt                    ! cutoff scaled to grid units

               IF ( (tt-t1 .GE. 0.) .AND. (tt-t1 .LT. tcut) ) THEN
                  ttend = tf + (tt-t1)
                  CALL gausswt2_v0(zz, ttend, xx, val, grid, wate, nz,
     .                          nt, nx, z1, t1, x1, zf, tf, xf, zsc,
     .                          tsc, xsc, cutoff, iwflag, nz, nt)
               ENDIF

               IF ( (tf-tt .LT. 0.) .AND. (tf-tt .LT. tcut) ) THEN
                  ttbeg = z1 - (zf-tt)
                  CALL gausswt2_v0(zz, ttbeg, xx, val, grid, wate, nz,
     .                          nt, nx, z1, t1, x1, zf, tf, xf, zsc,
     .                          tsc, xsc, cutoff, iwflag, nz, nt)
               ENDIF
            ENDIF

 700     CONTINUE

*  Put gridded fcn into result variable, dividing by summed weights. (as in
*  gaussfin, but indices needn't start at 1)

         l3 = 1
         DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

            k3 = 1
            DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               IF ( wate(k3,l3) .GT. 0. ) THEN
                  result(i,j,k,l,m,n) = grid(k3,l3) / wate(k3,l3)
               ELSE
                  result(i,j,k,l,m,n) = bad_flag_result
               ENDIF

               k3 = k3 + 1
 300        CONTINUE

            l3 = l3 + 1
 400     CONTINUE

         i3 = i3 + arg_incr(X_AXIS,ARG3)
 100  CONTINUE

         j3 = j3 + arg_incr(Y_AXIS,ARG3)
 200  CONTINUE

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
