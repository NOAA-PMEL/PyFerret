*
*  scat2gridlaplace_zt.F
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
* Nov 12, 2002 from scat2gridlaplace_xy
*
* Returns variable interpolated onto an equally-spaced Z-T grid.  
* Input is scattered triples: (z, t, f(x,t)); f may be function of x,y  
* Output is gridded data in z, t.  Calls routine "zgrid".
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


      SUBROUTINE  scat2gridlaplace_zt_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*100 buff


      WRITE (buff, 10)
   10 FORMAT ('Use Laplace weighting to grid scattered data to a ',
     .        'ZT grid.')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 7)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .      IMPLIED_BY_ARGS,  IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 6)

* Horizontal grid is determined by arguments 4 and 5, the result's x and y axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'ZPTS')
      CALL ef_set_arg_desc(id, arg, 
     .'x-coordinates of scattered input triples')
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
     .                    'Z axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'TAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'T axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'CAY')
      CALL ef_set_arg_desc(id, arg, 
     .     'Interpolation parameter: CAY')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'NRNG')
      CALL ef_set_arg_desc(id, arg, 
     .     'Interpolation parameter: NRNG')
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
      SUBROUTINE scat2gridlaplace_zt_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER nzout, ntout, nz2, nt2, nztgrid
      INTEGER nxin, nyin, nzin, ntin, nin
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      INTEGER nrng
      REAL value

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      CALL ef_get_one_val(id, 7, value)   ! nrng: extend axes for modulo axis case
      nrng = value

      nxin = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      nyin = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)
      nzin = 1 + arg_hi_ss(Z_AXIS,ARG1) - arg_lo_ss(Z_AXIS,ARG1)
      ntin = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)
      nin = 3* max(nzin,nyin,nzin,ntin)

      nzout = 1 + arg_hi_ss(Z_AXIS,ARG4) - arg_lo_ss(Z_AXIS,ARG4)
      ntout = 1 + arg_hi_ss(T_AXIS,ARG5) - arg_lo_ss(T_AXIS,ARG5)

      nz2 = nzout* 2
      nt2 = ntout* 2

      nzout = 2*nrng + nzout
      ntout = 2*nrng + ntout
      nztgrid = nzout* ntout

* zax  output z axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nz2, 1, 1, 1)

* tax  output t axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, nt2, 1, 1, 1)

* zgridded  work array - gridded data. 
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, nztgrid, 1, 1, 1)

* zscat - scattered points with gaps removed.
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, nin, 1, 1, 1)

* tscat - scattered points with gaps removed.
      CALL ef_set_work_array_dims (id, 5, 1, 1, 1, 1, nin, 1, 1, 1)

* fscat - scattered points with gaps removed.
      CALL ef_set_work_array_dims (id, 6, 1, 1, 1, 1, nin, 1, 1, 1)

************************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE  scat2gridlaplace_zt_compute(id, arg_1, arg_2, 
     .     arg_3, arg_4, arg_5, arg_6, arg_7, result, zax, tax,
     .     zgridded, zscat, tscat, fscat)

*  arg_1  zpts \
*  arg_2  tpts  > scattered z,t,f(z,t) triples to be gridded. f can be fcn of x,y
*  arg_3  fpts /
*  arg_4  zaxis of new grid
*  arg_5  taxis of new grid
*  arg_6  interpolation parameter   cay
*  arg_7  interpolation parameter   nrng


      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

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
     .     mem6lot:mem6hit)
      REAL arg_7(mem6lox:mem6hix, mem6loy:mem6hiy, mem6loz:mem6hiz, 
     .     mem6lot:mem6hit)
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

      INTEGER i, j, k, l, m
      INTEGER i1, i2, i4
      INTEGER i1n, i2n, i4n, j5, j5n
      INTEGER nzsize, ntsize
      INTEGER i3, j3, k3, l3
      INTEGER kz, lz, kl_index

      INTEGER nz, nt, nzpts, ntpts, nscat
      REAL z1, t1, dz, dt

      REAL cay, big
      INTEGER nrng
      PARAMETER (big = 1.e+35)	! missing-data flag expected by zgrid

*  Dimension the work arrays

      REAL*8 zax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 tax(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL zgridded(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL zscat(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)
      REAL tscat(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .               wrk5loz:wrk5hiz, wrk5lot:wrk5hit)
      REAL fscat(wrk6lox:wrk6hix, wrk6loy:wrk6hiy,
     .               wrk6loz:wrk6hiz, wrk6lot:wrk6hit)

      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), moduloz(4), modulot(4), regular(4)

*  Check to see if output axes are modulo

      CALL ef_get_axis_info (id, 4, ax_name, ax_units, backward, 
     .                       moduloz, regular)
      CALL ef_get_axis_info (id, 5, ax_name, ax_units, backward, 
     .                       modulot, regular)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, Z or T axis of the input arguments.

      nzpts = 0
      ntpts = 0

      DO 100 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG1)
            i1n = arg_hi_ss(m,ARG1)
            if (i1n-i1 .NE. 0) nzpts = 1 + (i1n - i1)
         ENDIF
 100  CONTINUE

      DO 110 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG2)
            i2n = arg_hi_ss(m,ARG2)
            if (i2n-i2 .NE. 0) ntpts = 1 + (i2n - i2)
         ENDIF
 110  CONTINUE

      IF (ntpts .NE. nzpts .OR. nzpts .EQ. 0) GOTO 900

      nscat = nzpts

*  Compute number of points in output axes.

      i4 = arg_lo_ss(Z_AXIS,ARG4)
      i4n = arg_hi_ss(Z_AXIS,ARG4)

      j5 = arg_lo_ss(T_AXIS,ARG5)
      j5n = arg_hi_ss(T_AXIS,ARG5)

      nz = 1 + (i4n - i4)
      nt = 1 + (j5n - j5)

*  Check that zax is an Z axis and tax a T axis

      IF (i4 .EQ. ef_unspecified_int4) then
         WRITE (errtxt, *) 'fourth argument must be a Z axis'
         GO TO 999
      ENDIF
      IF (j5 .EQ. ef_unspecified_int4) then
         WRITE (errtxt, *) 'fifth argument must be a T axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axes.

      call ef_get_coordinates(id, ARG4, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, ARG4), arg_hi_ss(Z_AXIS, ARG4), zax)

      call ef_get_coordinates(id, ARG5, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG5), arg_hi_ss(T_AXIS, ARG5), tax)

*  Set start and delta for output axes.

      z1 = zax(1,1,1,1)
      t1 = tax(1,1,1,1)

      dz = zax(2,1,1,1) - zax(1,1,1,1)
      dt = tax(2,1,1,1) - tax(1,1,1,1)

*  Get interpolation parameters.

      cay = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6), 
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6))
      nrng = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7), 
     .            arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7)) 

      IF (cay .LT. 0.) GOTO 910
      IF (nrng .LE. 0) GOTO 920

*  Compute result at each X, and each Y

      i3 = arg_lo_ss(X_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)
      DO 510 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         j3 = arg_lo_ss(Y_AXIS,ARG3)
         DO 500 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

*  If there is a missing value routine zgrid does not grid the data in a zone 
*  around that point, even if there are other nearby good data points.
*  Remove missing/bad data from the input (x,y,z) points.  This resets nscat.

         nscat = nzpts
         CALL nobadzt (arg_1, arg_2, arg_3, bad_flag(ARG3), 
     .                 nscat, i3, j3, k3, l3, zscat, tscat, fscat)

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered points.

         IF (moduloz(3)) CALL modscat (zax, nz, nscat, zscat)
         IF (modulot(4)) CALL modscat (tax, nt, nscat, tscat)

*  If an output axis is modulo, mirror the scattered points within NRNG
*  of each end to the other end of the region.

         IF (moduloz(3)) THEN
            CALL copyscat (zax, nz, nrng, nscat, zscat, tscat, fscat)
            z1 = z1 - float(nrng)* dz
            nz = nz + 2* nrng
         ENDIF

         IF (modulot(4)) THEN
            CALL copyscat (tax, nt, nrng, nscat, tscat, zscat, fscat)
            t1 = t1 - float(nrng)* dt
            nt = nt + 2* nrng
         ENDIF

*  Initialize result variable to 0.

        DO 200 k = 1, nz*nt
            zgridded(k,1,1,1) = 0.
 200     CONTINUE

*  Grid the data. 

          nzsize = nz
          ntsize = nt
          CALL zgrid (zgridded, nzsize, ntsize, nz, nt, z1, t1, 
     .              dz, dt, zscat, tscat, fscat, nscat, cay, nrng)

*  Put gridded z into result variable.  Use "bad_flag_result" rather than
*  "big" to mark bad data.

         kz = 1
         IF (moduloz(3)) kz = nrng + 1
         DO 410 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            lz = 1
            IF (modulot(4)) lz = nrng + 1
            DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
               kl_index = (lz-1 )* nz + kz

               IF (zgridded(kl_index,1,1,1) .EQ. big) THEN
                  result(i,j,k,l) = bad_flag_result
               ELSE
                  result(i,j,k,l) = zgridded(kl_index,1,1,1)
               ENDIF
               lz = lz + 1

  400       CONTINUE
            kz = kz + 1
  410    CONTINUE

         j3 = j3 + arg_incr(Y_AXIS,ARG3)
  500 CONTINUE
      i3 = i3 + arg_incr(X_AXIS,ARG3)
  510 CONTINUE

      RETURN

  900 CONTINUE

      IF (nzpts .NE. ntpts) THEN
         WRITE (errtxt,20) nzpts, ntpts
      ELSE IF (nzpts .EQ. 0) THEN
         WRITE (errtxt, 30) 
      ENDIF
      GOTO 999

  910 CONTINUE
      WRITE (errtxt,40)
      GOTO 999

  920 CONTINUE
      WRITE (errtxt,50)
      GOTO 999

  999 CALL EF_BAIL_OUT(id, errtxt)

   20 FORMAT('Input scattered z, t have different # of points', 2I8)

   30 FORMAT ('No data in scattered z, t points')

   40 FORMAT ('Gridding parameter CAY must be nonnegative Argument 6')

   50 FORMAT ('Gridding parameter NRNG must be positive. Argument 7')

      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END

      SUBROUTINE nobadzt (zin, tin, fin, badz, nscat, 
     .                    i3, j3, k3, l3, zscat, tscat, fscat)

* ACM 11/00 change so not exact comparison: wasnt catching all bad values.
* ACM  5/01 properly deal with f-scattered points which may be fcns of y and z

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      REAL zin(*), tin(*), badz
      REAL zscat(*), tscat(*), fscat(*)

      REAL fin(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .     mem3lot:mem3hit)

      INTEGER nscat, ngood, n, i3, j3, k3, l3
      REAL badcompare

      ngood = 0
      badcompare = abs(badz) - 1.

      IF (k3 .EQ. ef_unspecified_int4 ) THEN

         DO 100 n = 1, nscat
            IF (abs(fin(i3, j3, k3, n)) .LT. badcompare) THEN
               ngood = ngood + 1
               zscat(ngood) = zin(n)
               tscat(ngood) = tin(n)
               fscat(ngood) = fin(i3, j3, k3, n)
            ENDIF
  100    CONTINUE

      ELSE

         DO 200 n = 1, nscat
            IF (abs(fin(i3, j3, n, l3)) .LT. badcompare) THEN
               ngood = ngood + 1
               zscat(ngood) = zin(n)
               tscat(ngood) = tin(n)
               fscat(ngood) = fin(i3, j3, n, l3)
            ENDIF
  200    CONTINUE

      ENDIF
      nscat = ngood

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
