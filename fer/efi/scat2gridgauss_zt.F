*
*  scat2gridgauss_zt.F
*
*
*  This software was developed by the Thermal Modeling and Analysis11/13/02
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without anx fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and anx derivative works thereof, and
*  its supporting documentation for anx purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in anx publications that result from
*  the use of this software or in anx product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in anx advertising or publicity to endorse or promote anx products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with anx support, consulting,
*  training or assistance of anx kind with regard to the use, operation
*  and performance of this software nor to provide the user with anx
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND Anx EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR Anx SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR Anx DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
* Ansley Manke
* Nov 7, 2002 from scat2gridgauss_xy

* Returns variable interpolated onto an equally-spaced Z-T grid.  
* Input is scattered triples: (z, t, f(z,t)); f may be function of x, y
* and if it is a single list, it must be on a Z or T axis.  
* Output is gridded data in z, t.  Calls routine "gausswt2".
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


      SUBROUTINE  scat2gridgauss_zt_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Use Gaussian weighting to grid scattered data to a ',
     .        'ZT grid.')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .      IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

* YT grid is determined by arguments 4 and 5, the result's y and t axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'ZPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Z coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'T coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      WRITE (buff,20) 
   20 FORMAT ('F(Z,T) 3rd component of scattered input triples. ',
     .  'May be fcn of X and/or Y')
      CALL ef_set_arg_desc(id, arg, buff)
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'ZAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'Z axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'TAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'T axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'ZSCALE')
      WRITE (buff, 30)
      CALL ef_set_arg_desc(id, arg, buff)
   30 FORMAT 
     . ('mapping scale in Z direction, data units (e.g. m or km)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'TSCALE')
      WRITE (buff, 40)
      CALL ef_set_arg_desc(id, arg, buff)
   40 FORMAT 
     .('mapping scale in T direction, in data units (e.g.hours, days)')
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
      SUBROUTINE scat2gridgauss_zt_work_size(id)

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
      INTEGER nzout, ntout, nz2, nt2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nzout = 1 + arg_hi_ss(Z_AXIS,ARG4) - arg_lo_ss(Z_AXIS,ARG4)
      ntout = 1 + arg_hi_ss(T_AXIS,ARG5) - arg_lo_ss(T_AXIS,ARG5)

      nz2 = nzout* 2
      nt2 = ntout* 2

* zax  output z axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nz2, 1, 1, 1)

* tax  output t axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, nt2, 1, 1, 1)

* grid  work array - gridded data. 
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, 
     .                             nzout, ntout, 1, 1)

* wate - weights.
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, 
     .                             nzout, ntout, 1, 1)

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE  scat2gridgauss_zt_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, result, 
     .     zax, tax, grid, wate)

*  arg_1  ypts \
*  arg_2  tpts  > Scattered z,t,f(z,t) triples to be gridded.
*  arg_3  fpts /    fpts can be fcn of x,y
*  arg_4  z axis of output grid
*  arg_5  t axis of output grid
*  arg_6  interpolation parameter   yscale
*  arg_7  interpolation parameter   tscale
*  arg_8  interpolation parameter   cutoff
*  arg_9  not used

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
      INTEGER i1, i2, k1, L1
      INTEGER i3, j3, k3, l3
      INTEGER i4, i4n, j5, j5n

      INTEGER nzpts, ntpts, nscat
      INTEGER nz, nt, nzaxis, ntaxis
      INTEGER nzsize, ntsize
      INTEGER i1n, i2n

      REAL x1, z1, t1, xf, zf, tf
      REAL xx, zz, tt
      REAL xsc, zsc, tsc
      REAL cutoff
      REAL dz, dt, zcut, tcut, zzbeg, zzend, ttbeg, ttend

      REAL val
      INTEGER iwflag

*  Dimension the work arrays

      REAL*8 zax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 tax(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL grid(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL wate(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)

      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), moduloz(4), modulot(4), regularz(4),
     .        regulart(4)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Check to see if output axes are regular, and get modulo characteristics

      CALL ef_get_axis_info (id, 4, ax_name, ax_units, backward, 
     .                       moduloz, regularz)
      CALL ef_get_axis_info (id, 5, ax_name, ax_units, backward, 
     .                       modulot, regulart)

      IF (.NOT. regularz(3)) then
         WRITE (errtxt, *) 'output axes must be REGULARLY spaced'
         GO TO 999
      ENDIF
      IF (.NOT. regulart(2)) then
         WRITE (errtxt, *) 'output axes must be REGULARLY spaced'
         GO TO 999
      ENDIF

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, Z, or T axis of the input arguments.

      nzpts = 0
      ntpts = 0

      DO 100 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG1)
            i1n = arg_hi_ss(m,ARG1)
            if (i1n-i1 .GE. 0) nzpts = 1 + (i1n - i1)
         ENDIF
 100  CONTINUE

      DO 110 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG2)
            i2n = arg_hi_ss(m,ARG2)
            if (i2n-i2 .GE. 0) ntpts = 1 + (i2n - i2)
         ENDIF
 110  CONTINUE

      IF (nzpts .NE. ntpts .OR. nzpts .EQ. 0) GOTO 900

      nscat = nzpts

*  Compute number of points in output axes.

      i4 = arg_lo_ss(Z_AXIS,ARG4)
      i4n = arg_hi_ss(Z_AXIS,ARG4)

      j5 = arg_lo_ss(T_AXIS,ARG5)
      j5n = arg_hi_ss(T_AXIS,ARG5)

      nz = 1 + (i4n - i4)
      nt = 1 + (j5n - j5)
      nzsize = nz
      ntsize = nt

*  Check that zax is a Z axis and tax a T axis

      IF (i4 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'fourth argument must be a Z axis'
         GO TO 999
      ENDIF
      IF (j5 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'fifth argument must be a T axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axes.

      call ef_get_coordinates(id, ARG4, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, ARG4), arg_hi_ss(Z_AXIS, ARG4), zax)

      call ef_get_coordinates(id, ARG5, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG5), arg_hi_ss(T_AXIS, ARG5), tax)

*  Set start, end, and delta for output axes. 

      z1 = zax(1,1,1,1)
      t1 = tax(1,1,1,1)

      zf = zax(nz,1,1,1)
      tf = tax(nt,1,1,1)

*  X parameters for subroutine gausswt.  Calling with 1 x value

      nm = 1
      xx = 1.
      x1 = 1.
      xf = 1.
      xsc = 1. 

*  iwflag=1 for time wrapping; 0 for no wrapping

      iwflag = 0

*  Get interpolation parameters: mapping scales (data units)

      zsc  = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6), 
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6))

      tsc  = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7), 
     .            arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7))

      IF (zsc .LE. 0.) GOTO 910
      IF (tsc .LE. 0.) GOTO 920

*  And cutoff parameters:

      cutoff  = arg_8(arg_lo_ss(X_AXIS,ARG8), arg_lo_ss(Y_AXIS,ARG8), 
     .            arg_lo_ss(Z_AXIS,ARG8), arg_lo_ss(T_AXIS,ARG8))

      IF (cutoff .LE. 0.) GOTO 930

      nzaxis = arg_hi_ss(Z_AXIS,ARG4) - arg_lo_ss(Z_AXIS,ARG4) + 1
      ntaxis = arg_hi_ss(T_AXIS,ARG5) - arg_lo_ss(T_AXIS,ARG5) + 1

*  Compute result at each X, and each Y

      i3 = arg_lo_ss(X_AXIS,ARG3)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)

      DO 510 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      DO 500 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

*  Initialize sums of values and weights.

         DO l = 1, ntaxis
            DO k = 1, nzaxis
               grid(k,L,1,1) = 0.
            ENDDO
         ENDDO
  
         DO l = 1, ntaxis
            DO k = 1, nzaxis
               wate(k,L,1,1) = 0.
            ENDDO
         ENDDO

*  Loop over z and t, compute the weighted sums for gaussian-weighted mapping
*  onto the grid.  Args 1-2 may be on the X,Y,Z or T axis of ARG1 and
*  ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

         DO 300 n = 1, nscat

            CALL pickout2 (arg_1, arg_2, n, zz, tt)
	    IF (zz .NE. bad_flag(ARG1) .AND. tt .NE. bad_flag(ARG2)) THEN

            IF (arg_lo_ss(Z_AXIS,ARG3) .NE. ef_unspecified_int4)THEN
               val = arg_3(i3,j3,n,l3)
            ELSEIF (arg_lo_ss(T_AXIS,ARG3) .NE. ef_unspecified_int4)THEN
               val = arg_3(i3,j3,k3,n)
            ENDIF

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.

            IF (moduloz(3)) CALL modscat (zax, nz, 1, zz)
            IF (modulot(4)) CALL modscat (tax, nt, 1, tt)

            IF ( val  .NE. bad_flag(ARG3) ) THEN

               CALL gausswt2 (zz, tt, xx, val, grid, wate, nz, nt, nm, 
     .                    z1, t1, x1, zf, tf, xf, zsc, tsc, xsc, 
     .                    cutoff, iwflag, nzaxis, ntaxis)

C ACM modulo 11/9/00  Put points within cutoff of the end just beyond the
C                     other end, and use in the gridding computation.

               IF (moduloz(3)) THEN
                  dz = zf - z1
                  IF (nz .GT. 1) dz = (zf-z1)/real(nz-1)  ! gridbox size in data units
                  zcut = cutoff*zsc/dz		          ! cutoff scaled to grid units
                     
                  IF ((zz-z1 .GE. 0.) .AND. (zz-z1 .LT. zcut) ) THEN
                     zzend = zf + (zz-z1)

                     CALL gausswt2(zzend, tt, xx, val, grid, wate, nz, 
     .                    nt, nm, z1, t1, x1, zf, tf, xf, zsc, tsc, xsc, 
     .                    cutoff, iwflag, nzaxis, ntaxis)
                  ENDIF
                     
                  IF ((zf-zz .GE. 0.)  .AND. (zf-zz .LT. zcut) ) THEN
                     zzbeg = z1 - (zf-zz)

                     CALL gausswt2(zzbeg, tt, xx, val, grid, wate, nz, 
     .                    nt, nm, z1, t1, x1, zf, tf, xf, zsc, tsc, xsc, 
     .                    cutoff, iwflag, nzaxis, ntaxis)
                  ENDIF
               ENDIF

               IF (modulot(4)) THEN
                  dt = tf - t1
                  IF (nt .GT. 1) dt = (tf-t1)/real(nt-1)  ! gridbox size in data units
                  tcut = cutoff*tsc/dt		          ! cutoff scaled to grid units

                  IF ((tt-t1 .GE. 0.0) .AND.  (tt-t1  .LT. tcut) ) THEN
                     ttend = tf + (tt-t1)

                     CALL gausswt2(zz, ttend, xx, val, grid, wate, nz, 
     .                    nt, nm, z1, t1, x1, zf, tf, xf, zsc, tsc, xsc, 
     .                    cutoff, iwflag, nzaxis, ntaxis)
                  ENDIF
                     
                  IF ((tf-tt .LT. 0.)  .AND.  (tf-tt .LT. tcut) ) THEN

                     ttbeg = z1 - (zf-tt)
                     CALL gausswt2(zz, ttbeg, xx, val, grid, wate, nz, 
     .                    nt, nm, z1, t1, x1, zf, tf, xf, zsc, tsc, xsc, 
     .                    cutoff, iwflag, nzaxis, ntaxis)
                  ENDIF
               ENDIF
            ENDIF

            ENDIF  ! zz and tt not bad
 300     CONTINUE

*  Put gridded fcn into result variable, dividing by summed weights. (as in
*  gaussfin, but indices needn't start at 1)

         k1 = 1
         DO 410 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

         L1 = 1
         DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

            result(i,j,k,l) = bad_flag_result
            IF ( wate(k1,L1,1,1) .gt. 0.) THEN
               result(i,j,k,l) = grid(k1,L1,1,1)/ wate(k1,L1,1,1)
            ELSE
               result(i,j,k,l) = bad_flag_result
            ENDIF

            L1 = L1 + 1
 400        CONTINUE
         k1 = k1 + 1
 410     CONTINUE

      k3 = k3 + arg_incr(Y_AXIS,ARG3)
 500  CONTINUE
      i3 = i3 + arg_incr(X_AXIS,ARG3)
 510  CONTINUE

      RETURN


900   CONTINUE

      IF (nzpts .NE. ntpts) THEN
         WRITE (errtxt,20) nzpts, ntpts
      ELSE IF (nzpts .EQ. 0) THEN
         WRITE (errtxt, 30)
      ENDIF

      GOTO 999 

 910  CONTINUE
      WRITE (errtxt,40) 'Z', 6
      GOTO 999

 920  CONTINUE
      WRITE (errtxt,40) 'T', 7
      GOTO 999

 930  CONTINUE
      write (errtxt, 50) 8
      GOTO 999

 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN


   20 FORMAT ('Input scattered z, t have different # of points', 2i8)

   30 FORMAT ('No data in scattered z, t points')

   40 FORMAT (a1,
     .    ' Mapping scale parameter must be positive. Argument', I2)

   50 FORMAT ('Cutoff parameter must be positive. Argument', I2)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      END
