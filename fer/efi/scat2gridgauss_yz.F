*
*  scat2gridgauss_yz.F
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
* July 28 1998
*  revised 2/00 to use work arrays
* June 2000 ACM pass 4 gridding arguments:  y and z radius of influence,
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
*    5/2001    Let F(y,z) be a function of X and/or T
*
*  11/12/2002  Remove option 2) above: the spatial location of the scattered points
*              is a 1-D list.

* Returns variable interpolated onto an equally-spaced X-Y grid.  
* Input is scattered triples: (x, y, f(x,y)); may be functions of time.  
* Output is gridded data in x, y and time.  Calls routine "gausswt2".
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


      SUBROUTINE  scat2gridgauss_yz_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Use Gaussian weighting to grid scattered data to a ',
     .        'YZ grid.')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .      IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

* Horizontal grid is determined by arguments 4 and 5, the result's x and y axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Y coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'ZPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Z coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      WRITE (buff,20) 
   20 FORMAT ('F(X,T) 3rd component of scattered input triples. ',
     .  'May be fcn of X and/or T')
      CALL ef_set_arg_desc(id, arg, buff)
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'YAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'Y axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, YES, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'ZAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'Z axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'YSCALE')
      WRITE (buff, 30)
      CALL ef_set_arg_desc(id, arg, buff)
   30 FORMAT 
     .  ('Mapping scale in Y direction, data units (e.g. km or lat)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'ZSCALE')
      WRITE (buff, 40)
      CALL ef_set_arg_desc(id, arg, buff)
   40 FORMAT (
     .  'Mapping scale in Z direction, data units (e.g. m or km)')
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
      SUBROUTINE scat2gridgauss_yz_work_size(id)

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
      INTEGER nyout, nzout, ny2, nz2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nyout = 1 + arg_hi_ss(Y_AXIS,ARG4) - arg_lo_ss(Y_AXIS,ARG4)
      nzout = 1 + arg_hi_ss(Z_AXIS,ARG5) - arg_lo_ss(Z_AXIS,ARG5)

      ny2 = nyout* 2
      nz2 = nzout* 2

* yax  output y axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, ny2, 1, 1, 1)

* zax  output z axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, nz2, 1, 1, 1)

* grid  work array - gridded data. 
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, 
     .                             nyout, nzout, 1, 1)

* wate - weights.
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, 
     .                             nyout, nzout, 1, 1)

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE scat2gridgauss_yz_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, result, 

     .     yax, zax, grid, wate)

*  arg_1  ypts \
*  arg_2  zpts  > Scattered y,z,f(y,z) triples to be gridded.
*  arg_3  fpts /    F can be fcn of x,t
*  arg_4  yaxis of new grid
*  arg_5  zaxis of new grid
*  arg_6  interpolation parameter   yscale
*  arg_7  interpolation parameter   zscale
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
      INTEGER j4, j4n, k5, k5n

      INTEGER nypts, nzpts, nscat
      INTEGER ny, nz, nyaxis, nzaxis
      INTEGER nysize, nzsize
      INTEGER i3, j3, k3, l3
      INTEGER i1, i1n, i2, i2n, j1, k1

      REAL y1, z1, yf, zf, t1, tf
      REAL yy, zz, tt
      REAL ysc, zsc, tsc
      REAL cutoff
      REAL val
      INTEGER iwflag

*  Dimension the work arrays

      REAL*8 yax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 zax(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL grid(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL wate(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)

      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), moduloy(4), moduloz(4), regulary(4),
     .        regularz(4)

      REAL dy, dz, ycut, zcut, yybeg, yyend, zzbeg, zzend


      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Check to see if output axes are regular, and get modulo characteristics

      CALL ef_get_axis_info (id, 4, ax_name, ax_units, backward, 
     .                       moduloy, regulary)
      CALL ef_get_axis_info (id, 5, ax_name, ax_units, backward, 
     .                       moduloz, regularz)
     
      IF (.NOT. regulary(2)) then
         WRITE (errtxt, *) 'output axes must be REGULARLY spaced'
         GO TO 999
      ENDIF
      IF (.NOT. regularz(3)) then
         WRITE (errtxt, *) 'output axes must be REGULARLY spaced'
         GO TO 999
      ENDIF

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, or Z axis of the input arguments.

      nypts = 0
      nzpts = 0

      DO 100 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG1)
            i1n = arg_hi_ss(m,ARG1)
            if (i1n-i1 .NE. 0) nypts = 1 + (i1n - i1)
         ENDIF
 100  CONTINUE

      DO 110 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG2)
            i2n = arg_hi_ss(m,ARG2)
            if (i2n-i2 .NE. 0) nzpts = 1 + (i2n - i2)
         ENDIF
 110  CONTINUE

      IF (nypts .NE. nzpts .OR. nypts .EQ. 0) GOTO 900

      nscat = nypts

*  Compute number of points in output axes.

      j4 = arg_lo_ss(Y_AXIS,ARG4)
      j4n = arg_hi_ss(Y_AXIS,ARG4)

      k5 = arg_lo_ss(Z_AXIS,ARG5)
      k5n = arg_hi_ss(Z_AXIS,ARG5)

      ny = 1 + (j4n - j4)
      nz = 1 + (k5n - k5)
      nysize = ny
      nzsize = nz

*  Check that yax is a Y axis and zax a Z axis

      IF (j4 .EQ. ef_unspecified_int4) then
         WRITE (errtxt, *) 'fourth argument must be a Y axis'
         GO TO 999
      ENDIF
      IF (k5 .EQ. ef_unspecified_int4) then
         WRITE (errtxt, *) 'fifth argument must be a Z axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axes.

      call ef_get_coordinates(id, ARG4, Y_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG4), arg_hi_ss(Y_AXIS, ARG4), yax)

      call ef_get_coordinates(id, ARG5, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, ARG5), arg_hi_ss(Z_AXIS, ARG5), zax)

*  Set start, end, and delta for output axes. 


      y1 = yax(1,1,1,1)
      z1 = zax(1,1,1,1)

      yf = yax(ny,1,1,1)
      zf = zax(nz,1,1,1)

*  Time parameters for subroutine gausswt.  Calling with 1 time.

      nm = 1
      tt = 1.
      t1 = 1.
      tf = 1.
      tsc = 1. 

*  iwflag=1 for time wrapping; 0 for no wrapping

      iwflag = 0

*  Get interpolation parameters: mapping scales (data units)

      ysc  = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6), 
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6))

      zsc  = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7), 
     .            arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7))

      IF (ysc .LE. 0.) GOTO 910
      IF (zsc .LE. 0.) GOTO 920

*  And cutoff parameters:

      cutoff  = arg_8(arg_lo_ss(X_AXIS,ARG8), arg_lo_ss(Y_AXIS,ARG8), 
     .            arg_lo_ss(Z_AXIS,ARG8), arg_lo_ss(T_AXIS,ARG8))

      IF (cutoff .LE. 0.) GOTO 930

      nyaxis = arg_hi_ss(Y_AXIS,ARG4) - arg_lo_ss(Y_AXIS,ARG4) + 1
      nzaxis = arg_hi_ss(Z_AXIS,ARG5) - arg_lo_ss(Z_AXIS,ARG5) + 1

*  Compute result at each time, and each x.

      j3 = arg_lo_ss(Y_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)
      DO 510 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
       i3 = arg_lo_ss(X_AXIS,ARG3)
       DO 500 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

*  Initialize sums of values and weights.

         DO k = 1, nzaxis
           DO j = 1, nyaxis
               grid(j,k,1,1) = 0.
            ENDDO
         ENDDO

         DO k = 1, nzaxis
            DO j = 1, nyaxis
               wate(j,k,1,1) = 0.
            ENDDO
         ENDDO

*  Loop over x and y, compute the weighted sums for gaussian-weighted mapping
*  onto the grid.  Lat and longitude may be on the X,Y or Z axis of ARG1 and
*  ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

         DO 300 n = 1, nscat

            CALL pickoutxyz (arg_1, arg_2, n, yy, zz)

            IF (arg_lo_ss(Y_AXIS,ARG3) .NE. ef_unspecified_int4) THEN
               val = arg_3(i3,n,k3,l3)
            ELSE IF (arg_lo_ss(Z_AXIS,ARG3) .NE. ef_unspecified_int4) THEN
               val = arg_3(i3,j3,n,l3)
            ENDIF

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.

            IF (moduloy(2)) CALL modscat (yax, ny, 1, yy)
            IF (moduloz(3)) CALL modscat (zax, nz, 1, zz)

            IF ( val  .NE. bad_flag(ARG3) ) THEN
               CALL gausswt2 (yy, zz, tt, val, grid, wate, ny, nz, nm, 
     .                    y1, z1, t1, yf, zf, tf, ysc, zsc, tsc, 
     .                    cutoff, iwflag, nyaxis, nzaxis)


C ACM modulo 11/9/00  Put points within cutoff of the end just beyond the
C                     other end, and use in the gridding computation.

               IF (moduloy(1)) THEN
                  dy = yf - y1
                  IF (ny .GT. 1) dy = (yf-y1)/real(ny-1)  ! gridbox size in data units
                  ycut = cutoff*ysc/dy		          ! cutoff scaled to grid units
                     
                  IF ((yy-y1 .GE. 0.) .AND. (yy-y1 .LT. ycut) ) THEN
                     yyend = yf + (yy-y1)
                     CALL gausswt2(yyend, zz, tt, val, grid, wate, ny, 
     .                    nz, nm, y1, z1, t1, yf, zf, tf, ysc, zsc, tsc, 
     .                    cutoff, iwflag, nyaxis, nzaxis)
                  ENDIF
                     
                  IF ((yf-yy .GE. 0.)  .AND. (yf-yy .LT. ycut) ) THEN
                     yybeg = y1 - (yf-yy)
                     CALL gausswt2(yybeg, zz, tt, val, grid, wate, ny, 
     .                    nz, nm, y1, z1, t1, yf, zf, tf, ysc, zsc, tsc, 
     .                    cutoff, iwflag, nyaxis, nzaxis)
                  ENDIF
               ENDIF

               IF (moduloz(2)) THEN
                  dz = zf - z1
                  IF (nz .GT. 1) dz = (zf-z1)/real(nz-1)  ! gridbox size in data units
                  zcut = cutoff*zsc/dz		          ! cutoff scaled to grid units

                  IF ((zz-z1 .GE. 0.0) .AND.  (zz-z1  .LT. zcut) ) THEN
                     zzend = zf + (zz-z1)
                     CALL gausswt2(yy, zzend, tt, val, grid, wate, ny, 
     .                    nz, nm, y1, z1, t1, yf, zf, tf, ysc, zsc, tsc, 
     .                    cutoff, iwflag, nyaxis, nzaxis)
                  ENDIF
                     
                  IF ((zf-zz .LT. 0.)  .AND.  (zf-zz .LT. zcut) ) THEN
                     zzbeg = z1 - (zf-zz)
                     CALL gausswt2(yy, zzbeg, tt, val, grid, wate, ny, 
     .                    nz, nm, y1, z1, t1, yf, zf, tf, ysc, zsc, tsc, 
     .                    cutoff, iwflag, nyaxis, nzaxis)
                  ENDIF
               ENDIF
            ENDIF


 300     CONTINUE

*  Put gridded z into result variable, dividing by summed weights. (as in
*  gaussfin, but indices needn't start at 1)

         j1 = 1
         DO 410 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
         k1 = 1
         DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            result(i,j,k,l) = bad_flag_result
            IF ( wate(j1,k1,1,1) .gt. 0.) THEN
               result(i,j,k,l) = grid(j1,k1,1,1)/ wate(j1,k1,1,1)
            ELSE
               result(i,j,k,l) = bad_flag_result
            ENDIF

            k1 = k1 + 1
 400        CONTINUE
         j1 = j1 + 1
 410     CONTINUE

      i3 = i3 + arg_incr(X_AXIS,ARG3)

 500  CONTINUE
      l3 = l3 + arg_incr(T_AXIS,ARG3)
 510  CONTINUE

      RETURN


900   CONTINUE

      IF (nypts .NE. nzpts) THEN
        WRITE (errtxt,20) nypts, nzpts
      ELSE IF (nypts .EQ. 0) THEN
        WRITE (errtxt, 30)
      ENDIF
      GOTO 999 

 910  CONTINUE
      WRITE (errtxt,40) 'Y', 6
      GOTO 999

 920  CONTINUE
      WRITE (errtxt,40) 'Z', 7
      GOTO 999

 930  CONTINUE
      write (errtxt, 50) 8
      GOTO 999

 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN


   20 FORMAT ('Input scattered y, z have different # of points', 2i8)

   30 FORMAT ('No data in scattered y, z points')

   40 FORMAT (a1,
     .    ' Mapping scale parameter must be positive. Argument', I2)

   50 FORMAT ('Cutoff parameter must be positive. Argument', I2)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      END
