*
* SCATTER2GRIDLAPLACE_XZ.F
*
* Ansley Manke
* July 28 1998
*
* Returns variable interpolated onto an equally-spaced X-Z grid.  
* input and output data in x, y, depth, and time.
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


      SUBROUTINE SCATTER2GRIDLAPLACE_XZ_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

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
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .      NORMAL, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

* X-Z is determined by arguments 4 and 5, the result's x and z axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'X')
      CALL ef_set_arg_desc(id, arg, 
     .     'X coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'Z')
      CALL ef_set_arg_desc(id, arg, 
     .     'Z coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      CALL ef_set_arg_desc(id, arg, 
     .     'Z = F(X,Y)   F(x,z)  of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAX')
      CALL ef_set_arg_desc(id, arg, 'X axis of output grid')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'ZAX')
      CALL ef_set_arg_desc(id, arg, 'Z axis of output grid')
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
* In this subroutine we compute the result
*
      SUBROUTINE SCATTER2GRIDLAPLACE_XZ_compute(id, arg_1, arg_2, 
     .     arg_3, arg_4, arg_5, arg_6, arg_7, result)

*  arg_1  xpts \
*  arg_2  ypts  } scattered x,z,f(x,z) triples to be regridded.  Can be fcns of y,t
*  arg_3  zpts /
*  arg_4  X axis of new grid
*  arg_5  Z axis of new grid
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
      INTEGER ii3, jj3, kk3

      INTEGER nx, nz, nxpts, nzpts, nscat
      INTEGER i1n, i1del, i2n, i2del
      INTEGER i4n, i4del, k5, k5n, k5del
      INTEGER nxsize, nzsize
      INTEGER l_inc, j_inc, iz
      INTEGER kz, ik_index
      REAL x1, z1, dx, dz

      REAL cay, big
      INTEGER nrng
      PARAMETER (big = 1.e+35)		! missing-data flag expected by zgrid

      INTEGER NAX, NWORK
      PARAMETER (NAX = 400, NWORK=NAX*NAX)
      REAL*8 xax(NAX), zax(NAX)

      REAL zgridded(NWORK)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, or Z axis of the input arguments.

      DO 100 m = X_AXIS, Z_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG1)
            i1n = arg_hi_ss(m,ARG1)
            i1del = arg_incr(m,ARG1)
            if (i1del .NE. 0) nxpts = (i1n - i1)/ i1del
         ENDIF
 100  CONTINUE

      DO 110 m = X_AXIS, Z_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG2)
            i2n = arg_hi_ss(m,ARG2)
            i2del = arg_incr(m,ARG2)
            if (i2del .NE. 0) nzpts = (i2n - i2)/ i2del
         ENDIF
 110  CONTINUE

      IF (nzpts .NE. nxpts) then
         WRITE (*,*) 'Input x, z  have differing numbers of points'
         WRITE (*,*) 'nxpts, nzpts ', nxpts, nzpts
         GOTO 999
      ENDIF

      nscat = nxpts

*  Compute number of points in output axes.

      i4 = arg_lo_ss(X_AXIS,ARG4)
      i4n = arg_hi_ss(X_AXIS,ARG4)
      i4del = arg_incr(X_AXIS,ARG4)

      k5 = arg_lo_ss(Z_AXIS,ARG5)
      k5n = arg_hi_ss(Z_AXIS,ARG5)
      k5del = arg_incr(Z_AXIS,ARG5)

      nx = 1 + (i4n - i4)/ i4del
      nz = 1 + (k5n - k5)/ k5del
      nxsize = nx
      nzsize = nz

      call ef_get_coordinates(id, ARG4, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG4), arg_hi_ss(X_AXIS, ARG4), xax)

      call ef_get_coordinates(id, ARG5, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, ARG5), arg_hi_ss(Z_AXIS, ARG5), zax)

*  Set start and delta for output axes.

      x1 = xax(1)
      z1 = zax(1)

      dx = xax(2) - xax(1)
      dz = zax(2) - zax(1)

*  Get interpolation parameters.

      cay = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6), 
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6))

      nrng = arg_7(arg_lo_ss(X_AXIS,ARG7), arg_lo_ss(Y_AXIS,ARG7), 
     .            arg_lo_ss(Z_AXIS,ARG7), arg_lo_ss(T_AXIS,ARG7)) 


*  Compute result at each time, and over range of y.

      l_inc = max(1,res_incr(T_AXIS))
      j_inc = max(1,res_incr(Y_AXIS))

      DO 510 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS), l_inc
         DO 500 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS), j_inc

*  If there is a missing value routine zgrid does not grid the data in a zone 
*  around that point, even if there are other nearby good data points.
*  Remove missing/bad data from the input (x,y,z) points.  This resets nscat.

         ii3 = arg_lo_ss(X_AXIS,ARG3)
         jj3 = arg_lo_ss(Y_AXIS,ARG3)
         kk3 = arg_lo_ss(Z_AXIS,ARG3)

         nscat = nxpts
         CALL nobad (arg_1, arg_2, arg_3(ii3,jj3,kk3,l), 
     .         bad_flag(ARG3), nscat)

*  Initialize result variable to 0.

         DO 200 i = 1, NWORK
            zgridded(i) = 0.
  200   CONTINUE

         CALL zgrid (zgridded, nxsize, nzsize, nx, nz, x1, z1, 
     .              dx, dz, arg_1, arg_2, arg_3(ii3,jj3,kk3,l), 
     .              nscat, cay, nrng)

*  Put gridded f(x,z) into result variable.  Use "bad_flag_result" rather than
*  "big" to mark bad data.


         iz = 1
         DO 410 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS), 
     .              res_incr(X_AXIS)
            kz = 1
            DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS), 
     .                 res_incr(Z_AXIS)

               ik_index = (kz-1)* nx + iz

               IF (zgridded(ik_index) .EQ. big) 
     .              zgridded(ik_index) = bad_flag_result

               result(i,j,k,l) = zgridded(ik_index)
               kz = kz + 1

  400       CONTINUE
            iz = iz + 1
  410    CONTINUE

 500  CONTINUE
 510  CONTINUE

      RETURN

999   CONTINUE
      WRITE (*, *) 'ERROR RETURN SCATTER2GRIDLAPLACE_XZ'

      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END


      SUBROUTINE nobad (xscat, yscat, zscat, badz, nscat)

*  If there is a missing value routine zgrid does not grid the data in a zone 
*  around that point, even if there are other nearby good data points.
*  Remove missing/bad data from the input (x,y,zscat) points.

      REAL xscat(*), yscat(*), zscat(*), badz
      INTEGER nscat, ngood, i

      ngood = 0

      DO 100 i = 1, nscat
         IF (zscat(i) .ne. badz) THEN
            ngood = ngood + 1
            xscat(ngood) = xscat(i)
            yscat(ngood) = yscat(i)
            zscat(ngood) = zscat(i)
         ENDIF
  100 CONTINUE

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
