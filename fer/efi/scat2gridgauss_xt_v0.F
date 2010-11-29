*
*  scat2gridgauss_xt_v0.F
*
*
*  This software was developed by the Thermal Modeling and Analysis
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
c V62 *ACM* 2/2009 Call gausswt2_v0, with the original treatment of the
c            cutoff parameter, for backwards compatibility.

* Returns variable interpolated onto an equally-spaced X-T grid.  
* Input is scattered triples: (x, t, f(x,t)); f may be function of y, z  
* Output is gridded data in x, t.  Calls routine "gausswt2_v0".
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


      SUBROUTINE  scat2gridgauss_xt_v0_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Use Gaussian weighting, grid scattered data to ',
     .        'XT grid v0 weights')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .      IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

* XT grid is determined by arguments 4 and 5, the result's x and t axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'X coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'T coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      WRITE (buff,20) 
   20 FORMAT ('F(X,T) 3rd component of scattered input triples. ',
     .  'May be fcn of Y and/or Z')
      CALL ef_set_arg_desc(id, arg, buff)
      CALL ef_set_axis_influence(id, arg, NO, YES, YES, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'X axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'TAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'T axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'XSCALE')
      WRITE (buff, 30)
      CALL ef_set_arg_desc(id, arg, buff)
   30 FORMAT 
     . ('mapping scale in X direction, data units (e.g. km or lon)')
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
      SUBROUTINE scat2gridgauss_xt_v0_work_size(id)

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
      INTEGER nxout, ntout, nx2, nt2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4)
      ntout = 1 + arg_hi_ss(T_AXIS,ARG5) - arg_lo_ss(T_AXIS,ARG5)

      nx2 = nxout* 2
      nt2 = ntout* 2

* xax  output x axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nx2, 1, 1, 1)

* tax  output t axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, nt2, 1, 1, 1)

* grid  work array - gridded data. 
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, 
     .                             nxout, ntout, 1, 1)

* wate - weights.
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, 
     .                             nxout, ntout, 1, 1)

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE scat2gridgauss_xt_v0_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, result, 
     .     xax, tax, grid, wate)

*  arg_1  xpts \
*  arg_2  tpts  > Scattered x,t,f(x,t) triples to be gridded.
*  arg_3  fpts /    fpts can be fcn of y, z
*  arg_4  xaxis of new grid
*  arg_5  taxis of new grid
*  arg_6  interpolation parameter   xscale
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
      INTEGER i1, i2, L1
      INTEGER i4, i4n, j5, j5n

      INTEGER nxpts, ntpts, nscat
      INTEGER nx, nt, nxaxis, ntaxis
      INTEGER nxsize, ntsize
      INTEGER i3, j3, k3, l3
      INTEGER i1n, i2n

      REAL x1, y1, xf, yf, t1, tf
      REAL xx, yy, tt
      REAL xsc, ysc, tsc
      REAL cutoff
      REAL dx, dt, xcut, tcut, xxbeg, xxend, ttbeg, ttend

      REAL val
      INTEGER iwflag

*  Dimension the work arrays

      REAL*8 xax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
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

      LOGICAL backward(4), modulox(4), modulot(4), regularx(4),
     .        regulart(4)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


*  Check to see if output axes are regular, and get modulo characteristics

      CALL ef_get_axis_info (id, 4, ax_name, ax_units, backward, 
     .                       modulox, regularx)
      CALL ef_get_axis_info (id, 5, ax_name, ax_units, backward, 
     .                       modulot, regulart)

      IF (.NOT. regularx(1)) then
         WRITE (errtxt, *) 'output axes must be REGULARLY spaced'
         GO TO 999
      ENDIF
      IF (.NOT. regulart(4)) then
         WRITE (errtxt, *) 'output axes must be REGULARLY spaced'
         GO TO 999
      ENDIF

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, Z, or T axis of the input arguments.

      nxpts = 0
      ntpts = 0

      DO 100 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG1)
            i1n = arg_hi_ss(m,ARG1)
            if (i1n-i1 .GE. 0) nxpts = 1 + (i1n - i1)
         ENDIF
 100  CONTINUE

      DO 110 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG2)
            i2n = arg_hi_ss(m,ARG2)
            if (i2n-i2 .GE. 0) ntpts = 1 + (i2n - i2)
         ENDIF
 110  CONTINUE

      IF (nxpts .NE. ntpts .OR. nxpts .EQ. 0) GOTO 900

      nscat = nxpts

*  Compute number of points in output axes.

      i4 = arg_lo_ss(X_AXIS,ARG4)
      i4n = arg_hi_ss(X_AXIS,ARG4)

      j5 = arg_lo_ss(T_AXIS,ARG5)
      j5n = arg_hi_ss(T_AXIS,ARG5)

      nx = 1 + (i4n - i4)
      nt = 1 + (j5n - j5)
      nxsize = nx
      ntsize = nt

*  Check that xax is an X axis and tax a T axis

      IF (i4 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'fourth argument must be an X axis'
         GO TO 999
      ENDIF
      IF (j5 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'fifth argument must be a T axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axes.

      call ef_get_coordinates(id, ARG4, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG4), arg_hi_ss(X_AXIS, ARG4), xax)

      call ef_get_coordinates(id, ARG5, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG5), arg_hi_ss(T_AXIS, ARG5), tax)

*  Set start, end, and delta for output axes. 

      x1 = xax(1,1,1,1)
      t1 = tax(1,1,1,1)

      xf = xax(nx,1,1,1)
      tf = tax(nt,1,1,1)

*  Y parameters for subroutine gausswt.  Calling with 1 y value.

      nm = 1
      yy = 1.
      y1 = 1.
      yf = 1.
      ysc = 1. 

*  iwflag=1 for time wrapping; 0 for no wrapping

      iwflag = 0

*  Get interpolation parameters: mapping scales (data units)

      xsc  = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6), 
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6))

      tsc  = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7), 
     .            arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7))

      IF (xsc .LE. 0.) GOTO 910
      IF (tsc .LE. 0.) GOTO 920

*  And cutoff parameter:

      cutoff  = arg_8(arg_lo_ss(X_AXIS,ARG8), arg_lo_ss(Y_AXIS,ARG8), 
     .            arg_lo_ss(Z_AXIS,ARG8), arg_lo_ss(T_AXIS,ARG8))

      IF (cutoff .LE. 0.) GOTO 930

      nxaxis = arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4) + 1
      ntaxis = arg_hi_ss(T_AXIS,ARG5) - arg_lo_ss(T_AXIS,ARG5) + 1

*  Compute result at each Y, and each Z

      i3 = arg_lo_ss(X_AXIS,ARG3)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)

      DO 510 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      DO 500 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*  Initialize sums of values and weights.

         DO l = 1, ntaxis
            DO i = 1, nxaxis
               grid(i,L,1,1) = 0.
            ENDDO
         ENDDO
  
         DO l = 1, ntaxis
            DO i = 1, nxaxis
               wate(i,L,1,1) = 0.
            ENDDO
         ENDDO

*  Loop over x and t, compute the weighted sums for gaussian-weighted mapping
*  onto the grid.  Args 1-2 may be on the X,Y,Z or T axis of ARG1 and
*  ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

         DO 300 n = 1, nscat

            CALL pickout2 (arg_1, arg_2, n, xx, tt)

            IF (arg_lo_ss(X_AXIS,ARG3) .NE. ef_unspecified_int4)THEN
               val = arg_3(n,j3,k3,l3)
            ELSEIF (arg_lo_ss(Y_AXIS,ARG3) .NE. ef_unspecified_int4)THEN
               val = arg_3(i3,n,k3,l3)
            ELSEIF (arg_lo_ss(Z_AXIS,ARG3) .NE. ef_unspecified_int4)THEN
               val = arg_3(i3,j3,n,l3)
            ELSEIF (arg_lo_ss(T_AXIS,ARG3) .NE. ef_unspecified_int4)THEN
               val = arg_3(i3,j3,k3,n)
            ENDIF

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.

            IF (modulox(1)) CALL modscat (xax, nx, 1, xx)
            IF (modulot(4)) CALL modscat (tax, nt, 1, tt)

            IF ( val  .NE. bad_flag(ARG3) ) THEN
               CALL gausswt2_v0 (xx, tt, yy, val, grid, wate, nx, nt, nm, 
     .                    x1, t1, y1, xf, tf, yf, xsc, tsc, ysc, 
     .                    cutoff, iwflag, nxaxis, ntaxis)


C ACM modulo 11/9/00  Put points within cutoff of the end just beyond the
C                     other end, and use in the gridding computation.

               IF (modulox(1)) THEN
                  dx = xf - x1
                  IF (nx .GT. 1) dx = (xf-x1)/real(nx-1)  ! gridbox size in data units
                  xcut = cutoff*xsc/dx		          ! cutoff scaled to grid units
                     
                  IF ((xx-x1 .GE. 0.) .AND. (xx-x1 .LT. xcut) ) THEN
                     xxend = xf + (xx-x1)
                     CALL gausswt2_v0(xxend, tt, yy, val, grid, wate, nx, 
     .                    nt, nm, x1, t1, y1, xf, tf, yf, xsc, tsc, ysc, 
     .                    cutoff, iwflag, nxaxis, ntaxis)
                  ENDIF
                     
                  IF ((xf-xx .GE. 0.)  .AND. (xf-xx .LT. xcut) ) THEN
                     xxbeg = x1 - (xf-xx)
                     CALL gausswt2_v0(xxbeg, tt, yy, val, grid, wate, nx, 
     .                    nt, nm, x1, t1, y1, xf, tf, yf, xsc, tsc, ysc, 
     .                    cutoff, iwflag, nxaxis, ntaxis)
                  ENDIF
               ENDIF

               IF (modulot(4)) THEN
                  dt = tf - t1
                  IF (nt .GT. 1) dt = (tf-t1)/real(nt-1)  ! gridbox size in data units
                  tcut = cutoff*tsc/dt		          ! cutoff scaled to grid units

                  IF ((tt-t1 .GE. 0.0) .AND.  (tt-t1  .LT. tcut) ) THEN
                     ttend = tf + (tt-t1)
                     CALL gausswt2_v0(xx, ttend, yy, val, grid, wate, nx, 
     .                    nt, nm, x1, t1, y1, xf, tf, yf, xsc, tsc, ysc, 
     .                    cutoff, iwflag, nxaxis, ntaxis)
                  ENDIF
                     
                  IF ((tf-tt .LT. 0.)  .AND.  (tf-tt .LT. tcut) ) THEN
                     ttbeg = y1 - (yf-tt)
                     CALL gausswt2_v0(xx, ttbeg, yy, val, grid, wate, nx, 
     .                    nt, nm, x1, t1, y1, xf, tf, yf, xsc, tsc, ysc, 
     .                    cutoff, iwflag, nxaxis, ntaxis)
                  ENDIF
               ENDIF
            ENDIF


 300     CONTINUE

*  Put gridded fcn into result variable, dividing by summed weights. (as in
*  gaussfin, but indices needn't start at 1)

         i1 = 1
         DO 410 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         L1 = 1
         DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

            result(i,j,k,l) = bad_flag_result
            IF ( wate(i,L1,1,1) .GT. 0.) THEN
               result(i,j,k,l) = grid(i1,L1,1,1)/ wate(i1,L1,1,1)

            ELSE
               result(i,j,k,l) = bad_flag_result
            ENDIF

            L1 = L1 + 1
 400        CONTINUE
         i1 = i1 + 1
 410     CONTINUE

      k3 = k3 + arg_incr(Z_AXIS,ARG3)
 500  CONTINUE
      j3 = j3 + arg_incr(Y_AXIS,ARG3)
 510  CONTINUE

      RETURN


900   CONTINUE

      IF (nxpts .NE. ntpts) THEN
        WRITE (errtxt,20) nxpts, ntpts
      ELSE IF (nxpts .EQ. 0) THEN
        WRITE (errtxt, 30)
      ENDIF
      GOTO 999 

 910  CONTINUE
      WRITE (errtxt,40) 'X', 6
      GOTO 999

 920  CONTINUE
      WRITE (errtxt,40) 'T', 7
      GOTO 999

 930  CONTINUE
      write (errtxt, 50) 8
      GOTO 999

 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN


   20 FORMAT ('Input scattered x, t have different # of points', 2i8)

   30 FORMAT ('No data in scattered x, y points')

   40 FORMAT (a1,
     .    ' Mapping scale parameter must be positive. Argument', I2)

   50 FORMAT ('Cutoff parameter must be positive. Argument', I2)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      END
