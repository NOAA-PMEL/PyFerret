*
* scat2gridlaplace_xz.F
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
*
* Returns variable interpolated onto an equally-spaced X-Z grid.  
* input and output data in x, y, depth, and time.
*
*
* Nov 13, 2000 1) Allow modulo axes: if modulo take points from other end 
*                 to use in gridding each end
*              2) Check that the scattered points are listed on the I,J,or
*                 K axis only, as they may be functions of time.
*  (12/1/2000) 3) If the destination axis is modulo, treat the scattered
*                 points as modulo too.
*
*   12/7/2000  Add error checking on gridding parameters
*    5/2001    Let F(x,z) be a function of Y and/or T

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


      SUBROUTINE scat2gridlaplace_xz_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Use Laplace interpolation to grid scattered data to ',
     .        'an XZ grid.')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 7)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .       IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 6)

* X-Z is determined by arguments 4 and 5, the result's x and z axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 
     .'x-coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'ZPTS')
      CALL ef_set_arg_desc(id, arg, 
     .'z-coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      WRITE (fcn_desc, 20) 
   20 FORMAT ('F(X,Z) 3rd component of scattered input triples',
     .  ' may be fcn of Y and/or time')
      CALL ef_set_arg_desc(id, arg, fcn_desc)
      CALL ef_set_axis_influence(id, arg, NO, YES, NO, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg,'X axis coords, regular output grid')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'ZAXPTS')
      CALL ef_set_arg_desc(id, arg,'Z axis coords, regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, NO)

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

**                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END

*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE scat2gridlaplace_xz_work_size(id)

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
      INTEGER nxout, nzout, nx2, nz2, nxzgrid
      INTEGER nxin, nyin, nzin, ntin, nin
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      INTEGER nrng
      REAL value

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      CALL ef_get_one_val(id, 7, value)   ! nrng: extend axes for modulo axis case
      nrng = value

      nxin = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      nyin = 1 + arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)
      nzin = 1 + arg_hi_ss(Z_AXIS,ARG1) - arg_lo_ss(Z_AXIS,ARG1)
      ntin = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)
      nin = 3* max(nxin,nyin,nzin,ntin)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4)
      nzout = 1 + arg_hi_ss(Z_AXIS,ARG5) - arg_lo_ss(Z_AXIS,ARG5)

      nx2 = nxout* 2
      nz2 = nzout* 2

      nxout = 2*nrng + nxout
      nzout = 2*nrng + nzout
      nxzgrid = nxout* nzout

* xax  output x axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nx2, 1, 1, 1)

* zax  output z axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, nz2, 1, 1, 1)

* zgridded  work array - gridded data. 
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, nxzgrid, 1, 1, 1)

* xscat - scattered points with gaps removed.
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, nin, 1, 1, 1)

* zscat - scattered points with gaps removed.
      CALL ef_set_work_array_dims (id, 5, 1, 1, 1, 1, nin, 1, 1, 1)

* fscat - scattered points with gaps removed.
      CALL ef_set_work_array_dims (id, 6, 1, 1, 1, 1, nin, 1, 1, 1)

************************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE scat2gridlaplace_xz_compute(id, arg_1, arg_2, 
     .     arg_3, arg_4, arg_5, arg_6, arg_7, result, xax, zax,
     .     zgridded, xscat, zscat, fscat)

*  arg_1  xpts \
*  arg_2  zpts  } scattered x,z,f(x,z) triples to be regridded.  Can be fcns of t
*  arg_3  zpts /
*  arg_4  X axis of new grid
*  arg_5  Z axis of new grid
*  arg_6  interpolation parameter   cay
*  arg_7  interpolation parameter   nrng


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
      INTEGER i3, j3, k3, l3

      INTEGER nx, nz, nxpts, nzpts, nscat
      INTEGER i1n, i2n
      INTEGER i4n, k5, k5n
      INTEGER nxsize, nzsize
      INTEGER iz, kz, ik_index
      REAL x1, z1, dx, dz

      REAL cay, big
      INTEGER nrng
      PARAMETER (big = 1.e+35)	! missing-data flag expected by zgrid

*  Dimension the work arrays

      REAL*8 xax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 zax(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL zgridded(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL xscat(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)
      REAL zscat(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .               wrk5loz:wrk5hiz, wrk5lot:wrk5hit)
      REAL fscat(wrk6lox:wrk6hix, wrk6loy:wrk6hiy,
     .               wrk6loz:wrk6hiz, wrk6lot:wrk6hit)

      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulox(4), moduloz(4), regular

*  Check to see if output axes are modulo

      CALL ef_get_axis_info (id, 4, ax_name, ax_units, backward, 
     .                       modulox, regular)
      CALL ef_get_axis_info (id, 5, ax_name, ax_units, backward, 
     .                       moduloz, regular)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, or Z axis of the input arguments.

      nxpts = 0
      nzpts = 0

      DO 100 m = X_AXIS, Z_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG1)
            i1n = arg_hi_ss(m,ARG1)
            if (i1n-i1 .NE. 0) nxpts = i1n - i1 + 1
         ENDIF
 100  CONTINUE

      DO 110 m = X_AXIS, Z_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG2)
            i2n = arg_hi_ss(m,ARG2)
            if (i2n-i2 .NE. 0) nzpts = i2n - i2 + 1
         ENDIF
 110  CONTINUE

      IF (nzpts .NE. nxpts .OR. nxpts .eq. 0) GOTO 900

      nscat = nxpts

*  Compute number of points in output axes.

      i4 = arg_lo_ss(X_AXIS,ARG4)
      i4n = arg_hi_ss(X_AXIS,ARG4)

      k5 = arg_lo_ss(Z_AXIS,ARG5)
      k5n = arg_hi_ss(Z_AXIS,ARG5)

      nx = 1 + i4n - i4
      nz = 1 + k5n - k5

*  Check that xax is an X axis and zax a Z axis

      IF (i4 .EQ. ef_unspecified_int4) then
         WRITE (errtxt, *) 'fourth argument must be an X axis'
         GO TO 999
      ENDIF
      IF (k5 .EQ. ef_unspecified_int4) then
         WRITE (errtxt, *) 'fifth argument must be a Z axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axes.

      call ef_get_coordinates(id, ARG4, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG4), arg_hi_ss(X_AXIS, ARG4), xax)

      call ef_get_coordinates(id, ARG5, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, ARG5), arg_hi_ss(Z_AXIS, ARG5), zax)

*  Set start and delta for output axes.

      x1 = xax(1,1,1,1)
      z1 = zax(1,1,1,1)

      dx = xax(2,1,1,1) - xax(1,1,1,1)
      dz = zax(2,1,1,1) - zax(1,1,1,1)

*  Get interpolation parameters.

      cay = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6), 
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6))

      nrng = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7), 
     .            arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7)) 

      IF (cay .LT. 0.) GOTO 910
      IF (nrng .LE. 0) GOTO 920

*  Compute result at each time, and over range of y.


      i3 = arg_lo_ss(X_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)

      l3 = arg_lo_ss(T_AXIS,ARG3)
      DO 510 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         j3 = arg_lo_ss(Y_AXIS,ARG3)
         DO 500 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

*  If there is a missing value routine zgrid does not grid the data in a zone 
*  around that point, even if there are other nearby good data points.
*  Remove missing/bad data from the input (x,y,z) points.  This resets nscat.

         nscat = nxpts

         CALL nobadxz (arg_1, arg_2, arg_3, bad_flag(ARG3), 
     .                 nscat, i3, j3, k3, l3, xscat, zscat, fscat)

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered points.

         IF (modulox(1)) CALL modscat (xax, nx, nscat, xscat)
         IF (moduloz(3)) CALL modscat (zax, nz, nscat, zscat)

*  If an output axis is modulo, mirror the scattered points within NRNG
*  of each end to the other end of the region.

         IF (modulox(1)) THEN
            CALL copyscat (xax, nx, nrng, nscat, xscat, zscat, fscat)
            x1 = x1 - float(nrng)* dx
            nx = nx + 2* nrng
         ENDIF

         IF (moduloz(3)) THEN
            CALL copyscat (zax, nz, nrng, nscat, zscat, xscat, fscat)
            z1 = z1 - float(nrng)* dz
            nz = nz + 2* nrng
         ENDIF

*  Initialize result variable to 0.

         DO 200 i = 1, nx*nz
            zgridded(i,1,1,1) = 0.
  200    CONTINUE

         nxsize = nx
         nzsize = nz
         CALL zgrid (zgridded, nxsize, nzsize, nx, nz, x1, z1, 
     .              dx, dz, xscat, zscat, fscat, nscat, cay, nrng)

*  Put gridded f(x,z) into result variable.  Use "bad_flag_result"
*  rather than "big" to mark bad data.

         iz = 1
         IF (modulox(1)) iz = nrng + 1
         DO 410 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
            kz = 1
            IF (moduloz(3)) kz = nrng + 1
            DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               ik_index = (kz-1)* nx + iz

               IF (zgridded(ik_index,1,1,1) .EQ. big) 
     .              zgridded(ik_index,1,1,1) = bad_flag_result

               result(i,j,k,l) = zgridded(ik_index,1,1,1)
               kz = kz + 1

  400       CONTINUE
            iz = iz + 1
  410    CONTINUE

         j3 = j3 + arg_incr(Y_AXIS,ARG3)
 500  CONTINUE
      l3 = l3 + arg_incr(T_AXIS,ARG3)
 510  CONTINUE

      RETURN

900   CONTINUE

      IF (nxpts .NE. nzpts) THEN
         WRITE (errtxt,20) nxpts, nzpts
      ELSE IF (nxpts .EQ. 0) THEN
         WRITE (errtxt, 30) 
      ENDIF
      GOTO 999

  910 CONTINUE
      WRITE (errtxt,40)
      GOTO 999

  920 CONTINUE
      WRITE (errtxt,50)
      GOTO 999

 999  CALL EF_BAIL_OUT(id, errtxt)

   20 FORMAT('Input scattered x, z have different # of points', 2i8)

   30 FORMAT ('Scattered points must be indexed I,J,or L.  Use RESHAPE',
     .        ' to list them in I,J,or L.')

   40 FORMAT ('Gridding parameter CAY must be nonnegative Argument 6')

   50 FORMAT ('Gridding parameter NRNG must be positive. Argument 7')

      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END


      SUBROUTINE nobadxz (xin, zin, fin, badz, nscat, 
     .                    i3, j3, k3, l3, xscat, zscat, fscat)

* ACM 11/00 change so not exact comparison: wasnt catching all bad values.
* ACM  5/01 properly deal with f-scattered points which may be fcns of y and t

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      REAL xin(*), zin(*), badz
      REAL xscat(*), zscat(*), fscat(*)

      REAL fin(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .     mem3lot:mem3hit)

      INTEGER nscat, ngood, n, i3, j3, k3, l3
      REAL badcompare

      ngood = 0
      badcompare = abs(badz) - 1.

      IF (i3 .EQ. ef_unspecified_int4 ) THEN

         DO 100 n = 1, nscat
            IF (abs(fin(i3, j3, n, l3)) .LT. badcompare) THEN
               ngood = ngood + 1
               xscat(ngood) = xin(n)
               zscat(ngood) = zin(n)
               fscat(ngood) = fin(i3, j3, n, l3)
            ENDIF
  100    CONTINUE

      ELSE

         DO 200 n = 1, nscat
            IF (abs(fin(n, j3, k3, l3)) .LT. badcompare) THEN
               ngood = ngood + 1
               xscat(ngood) = xin(n)
               zscat(ngood) = zin(n)
               fscat(ngood) = fin(n, j3, k3, l3)
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
