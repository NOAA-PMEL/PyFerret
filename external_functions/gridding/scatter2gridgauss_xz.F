*
*  SCATTER2GRIDGAUSS_XZ.F
*
* Ansley Manke
* July 28 1998
*
* Returns variable interpolated onto an equally-spaced X-Y grid.  
* Input is scattered triples: (x, y, f(x,y)); may be functions of z and/or time.
* Output is gridded data in x, y, z, and time.  Calls routine "gausswt".
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


      SUBROUTINE  SCATTER2GRIDGAUSS_XZ_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Use Gaussian weighting to grid scattered data to an ',
     .        'XZ grid.')
      CALL ef_set_desc(id, fcn_desc)


      CALL ef_set_num_args(id, 6)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .      NORMAL, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

* XZ grid is determined by arguments 4 and 5, the result's x and z axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'X coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'ZPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Z coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      CALL ef_set_arg_desc(id, arg, 
     .     'F(X,Z)   3rd component of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAX')
      CALL ef_set_arg_desc(id, arg, 'X axis of output grid')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'ZAX')
      CALL ef_set_arg_desc(id, arg, 'Y axis of output grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'CUTOFF')
      CALL ef_set_arg_desc(id, arg, 
     .     'Interpolation parameter: cutoff limit')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE  SCATTER2GRIDGAUSS_XZ_compute(id, arg_1, arg_2, 
     .     arg_3, arg_4, arg_5, arg_6, result)



*  arg_1  xpts \
*  arg_2  ypts  > Scattered x,z,f(x,y) triples to be gridded.
*  arg_3  zpts /    Can be fcn of z,t
*  arg_4  X axis of new grid
*  arg_5  Z axis of new grid
*  arg_6  interpolation parameter   cutoff

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
      INTEGER i1, i2
      INTEGER i1n, i1del, i2n, i2del
      INTEGER nxpts, nzpts, nscat
      INTEGER i4, i4n, i4del, j5, j5n, j5del
      INTEGER nxsize, nzsize
      REAL xsc, zsc, tsc 
      REAL tt, t1, tf
      REAL iwflag, l3, ii3, jj3, kk3, ll3

      INTEGER nx, nz, nxaxis, nzaxis
      REAL x1, z1, xf, zf, dx, dz
      REAL xx, zz, val, cutoff

      REAL NAX
      PARAMETER (NAX = 400)
      REAL*8 xax(NAX), zax(NAX)

      REAL grid(NAX, NAX, 1), wate(NAX, NAX, 1)
      CHARACTER*100 errtxt

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
            if (i1del .NE. 0) nxpts = 1 + (i1n - i1)/ i1del
         ENDIF
 100  CONTINUE

      DO 110 m = X_AXIS, Z_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG2)
            i2n = arg_hi_ss(m,ARG2)
            i2del = arg_incr(m,ARG2)
            if (i2del .NE. 0) nzpts = 1 + (i2n - i2)/ i2del
         ENDIF
 110  CONTINUE

      IF (nzpts .NE. nxpts ) GOTO 999

      nscat = nxpts

*  Compute number of points in output axes.

      i4 = arg_lo_ss(X_AXIS,ARG4)
      i4n = arg_hi_ss(X_AXIS,ARG4)
      i4del = arg_incr(X_AXIS,ARG4)

      j5 = arg_lo_ss(Z_AXIS,ARG5)
      j5n = arg_hi_ss(Z_AXIS,ARG5)
      j5del = arg_incr(Z_AXIS,ARG5)

      nx = 1 + (i4n - i4)/ i4del
      nz = 1 + (j5n - j5)/ j5del
      nxsize = nx
      nzsize = nz

      call ef_get_coordinates(id, ARG4, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG4), arg_hi_ss(X_AXIS, ARG4), xax)

      call ef_get_coordinates(id, ARG5, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, ARG5), arg_hi_ss(Z_AXIS, ARG5), zax)

*  Set start, end, and delta for output axes. 
*  	x1,z1=west/top edge of grid (center of 1st box in data units)
*  	xf,zf=east/bottom edge of grid (center of final box)

      dx = xax(2) - xax(1)
      dz = zax(2) - zax(1)

      x1 = xax(1)
      z1 = zax(1)

      xf = xax(nx)
      zf = zax(nz)

*  mapping scales (data units)

      xsc = 1                
      zsc = 1.
      tsc = 1. 

*  Time parameters for subroutine gausswt.  Calling with 1 time.

      nm = 1
      tt = 1.
      t1 = 1.
      tf = 1.

*  iwflag=1 for time wrapping; 0 for no wrapping

      iwflag = 0

*  Get interpolation parameter.

      cutoff  = arg_6(arg_lo_ss(X_AXIS,ARG6), arg_lo_ss(Y_AXIS,ARG6), 
     .            arg_lo_ss(Z_AXIS,ARG6), arg_lo_ss(T_AXIS,ARG6))

      nxaxis = NAX
      nzaxis = NAX

*  Initialize sums of values and weights.
      DO j = 1, NAX
         DO i = 1, NAX
            grid(i,j,1) = 0.
         ENDDO
      ENDDO

      DO j = 1, NAX
         DO i = 1, NAX
            wate(i,j,1) = 0.
         ENDDO
      ENDDO

*  Compute result at each time, and each depth.

      l3 = arg_lo_ss(T_AXIS,ARG3)

      DO 510 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS), 
     .                       max(1,res_incr(T_AXIS))
         DO 500 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS), 
     .                       max(1,res_incr(Y_AXIS))

*  Loop over x and z, compute the weighted sums for gaussian-weighted mapping
*  onto the grid.  Lat and longitude may be on the X,Y or Z axis of ARG1 and
*  ARG2: send them to a subroutine which collapses the extra dimensions so the
*  data value can be found.


         ii3 = arg_lo_ss(X_AXIS,ARG3)
         jj3 = arg_lo_ss(Y_AXIS,ARG3)
         kk3 = arg_lo_ss(Z_AXIS,ARG3)
         ll3 = arg_lo_ss(T_AXIS,ARG3) - 1 + l
         if (arg_lo_ss(T_AXIS,ARG3) .eq. -111) ll3 = -111
         DO 300 n = 1, nscat

            CALL pickoutxyz (arg_1, arg_2, arg_3(ii3,jj3,kk3,ll3), 
     .                    n, xx, zz, val)

            IF ( val  .NE. bad_flag(ARG3) )
     .      CALL gausswt (xx, zz, tt, val, grid, wate, nx, nz, nm, 
     .                    x1, z1, t1, xf, zf, tf, xsc, zsc, tsc, 
     .                    iwflag, cutoff, nxaxis, nzaxis) 

 300     CONTINUE

*  Put gridded z into result variable, dividing by summed weights. (as in
*  gaussfin, but indices needn't start at 1)


         m = 1
         DO 410 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS), 
     .                       max(1,res_incr(X_AXIS))
         DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS), 
     .                       max(1,res_incr(Z_AXIS))

            result(i,j,k,l) = 0.
            IF ( wate(i,k,m) .gt. 0.) THEN
               result(i,j,k,l) = grid(i,k,m)/ wate(i,k,m)
            ELSE
               result(i,j,k,l) = bad_flag_result
            ENDIF

 400        CONTINUE
 410     CONTINUE

 500  CONTINUE
      l3 = l3 + arg_incr(T_AXIS,ARG3)
 510  CONTINUE

      RETURN

999   CONTINUE

   20 FORMAT('SCATTER2GRIDGAUSS_XZ; Input x, z have different ',
     .       '# of points', 2i8)

      WRITE (errtxt,20) nxpts, nzpts

      CALL EF_BAIL_OUT(id, errtxt)


      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END

      SUBROUTINE pickoutxyz (xpts, ypts, fcnpts, n, xx, yy, val)

*  Pick out nth item from xpts and ypts, which are really 4-dimensioned arrays
*  with normal axes on three of the four axes.

      REAL xpts(*), ypts(*), fcnpts(*)
      REAL xx, yy, val
      INTEGER n

      xx = xpts(n)
      yy = ypts(n)
      val = fcnpts(n)

      RETURN
      END


C  From Billy Kessler 25-Aug-1998

C  !ACM Modified to pass NAX, NAY dimensions of grid and wate.
C   and to deal with nx, ny, or nm = 1.  Also to send cutoff as a parameter.


c.............subroutines to do gaussian-weighted mapping onto grids.

c.......3sep97: rewritten to give each nested loop a separate loop number.
c		otherwise "if (delx.gt.xcut) go to 100" does not work right!
c************************************************************************
c............sub gausswt forms the weight sums inside loop over all data.
c............--->>> 3-d mapping (x,y,t)
c............--->>> allows wraparound year (flagged by arg iwflag)
c............method is to call this sub for each data value
c............  sub loops on grid locations, maps each irregular data point to
c............  all possible gridpts, weighting by 3-d distance from gridpt.
c............  all calcs done in gridbox units
c............xx/x1/xf/xsc all in same units
c............yy/y1/yf/ysc all in same units
c............tt/t1/tf/tsc all in same units (but nm can be anything)
c............   note that, ie, t1 is the center of gridbox 1. So if dates
c............   are in months, Jan 15=1, Dec 15=12, and Jan 1=0.5, Dec 31=12.5.

c  i	xx,yy,tt=x/y/t location of data pt (data units)
c  i	val=data value at this (xx,yy,tt)
c  o	grid(nx,ny,nm)=sum of weighted values
c  o	wate(nx,ny,nm)=sum of weights
c  i	nx,ny,nm=size of grids
c  i	x1,y1,t1=west/south/date edge of grid (center of 1st box in data units)
c  i	xf,yf,tf=east/north/date edge of grid (center of final box)
c  i	xsc,ysc,tsc=mapping scales (data units)
c  i	iwflag=1 for time wrapping; 0 for no wrapping
c--------------------------------------------------------------------------

	subroutine gausswt (xx,yy,tt,val,grid,wate,nx,ny,nm,x1,y1,t1,
     .                    xf,yf,tf,xsc,ysc,tsc,iwflag,cutoff,nax,nay)
 
	real grid(nax,nay,*), wate(nax,nay,*)

	dx=1.
	dy=1. 
	dt=1.
	if (nx .gt. 1) dx=(xf-x1)/real(nx-1)  ! gridbox sizes in data units
	if (ny .gt. 1) dy=(yf-y1)/real(ny-1) 
	if (nm .gt. 1) dt=(tf-t1)/real(nm-1)

	xxg=(xx-x1)/dx+1.		  ! grid values of data location
	yyg=(yy-y1)/dy+1.
	ttg=(tt-t1)/dt+1.

c	cutoff=2.			  ! cutoff limits search (min wt=e**-4)
	xcut=cutoff*xsc/dx		  ! cutoffs scaled to grid units
	ycut=cutoff*ysc/dy		  ! look only twice the scale width
	tcut=cutoff*tsc/dt		  !   from the gridbox center

	do 100 i=1,nx			  ! loop on x gridpoints
	xgp=real(i)			  ! center of gridbox
	delx=abs(xgp-xxg)		  ! distance of data pt from grid ctr
	if (delx.gt.xcut) go to 100  	  ! only do nearby points

	do 101 j=1,ny			  ! loop on y gridpoints, same procedure
	ygp=real(j)		
	dely=abs(ygp-yyg)
	if (dely.gt.ycut) go to 101

	do 102 m=1,nm			  ! loop on t gridpoints, same procedure
	tgp=real(m)
	delt=abs(tgp-ttg)
	if (delt.gt.tcut .and. iwflag.eq.1) 
     .		delt=abs(delt-real(nm)) 	! allow flagged time wrapping
	if (delt.gt.tcut) go to 102

	xgas=(delx*dx/xsc)**2		  	! make gaussian exponents
	ygas=(dely*dy/ysc)**2
	tgas=(delt*dt/tsc)**2
	expn=exp(-xgas-ygas-tgas)		! make the gaussian weight
	wate(i,j,m)=wate(i,j,m)+expn		! sum the weights
	grid(i,j,m)=grid(i,j,m)+val*expn	! sum the weighted values

102	continue
101	continue
100	continue

	return
	end

c************************************************************************
c............sub gaussfin divides weighted values by sum of weights
c............call this outside of summation loop.
c............for 2-d mapping, call gaussfin with nm=1.

	subroutine gaussfin (nx,ny,nm,grid,wate)

	real grid(nx,ny,nm),wate(nx,ny,nm)

        
	realbad=1.e35

	do 100 i=1,nx	! these extra loop number may not be necessary
	do 101 j=1,ny	! here, since there are no if-jumps and every
	do 102 m=1,nm	! element is goen through no matter what
	if (wate(i,j,m).gt.0.) then
		grid(i,j,m)=grid(i,j,m)/wate(i,j,m)
	else
		grid(i,j,m)=realbad
	endif
102	continue
101	continue
100	continue

	return
	end

c************************************************************************
c************************************************************************
c............sub gausswt-2d forms the weight sums inside loop over all data.
c............--->>> 2-d mapping (x,y)
c............--->>> allows wraparound year (flagged by arg iwflag) (use x -> t)
c............2-d also makes a grid of the number of obs/gridbox
c............method is to call this sub for each data value
c............  sub loops on grid locations, maps each irregular data point to
c............  all possible gridpts, weighting by 3-d distance from gridpt.
c............  all calcs done in gridbox units
c............xx/x1/xf/xsc all in same units
c............yy/y1/yf/ysc all in same units
c............after sum loop, use gaussfin to finish. Call gaussfin with nm=1.

c  i	xx,yy=x/y location of data pt (data units)
c  i	val=data value at this (xx,yy)
c  o	grid(nx,ny)=sum of weighted values
c  o	wate(nx,ny)=sum of weights
c  o	obs(nx,ny)=number of obs/box. See note.
c  i	nx,ny=size of grids
c  i	x1,y1=west/south edge of grid (center of 1st box in data units)
c  i	xf,yf=east/north edge of grid (center of final box)
c  i	xsc,ysc=mapping scales (data units)
c  i	iwflag=1 for time wrapping; 0 for no wrapping
c  Note: obs accumulates. If this sub is called repeatedly ==>> reset
c--------------------------------------------------------------------------

	subroutine gausswt_2d (xx,yy,val,grid,wate,obs,nx,ny,x1,y1,
     .			xf,yf,xsc,ysc,iwflag)

	real grid(nx,ny), wate(nx,ny)
	real obs(nx,ny)

	dx=(xf-x1)/real(nx-1)		  ! x-grid size in data units
	dy=(yf-y1)/real(ny-1) 

	xxg=(xx-x1)/dx+1.		  ! grid values of data location
	yyg=(yy-y1)/dy+1.

c.................save the number of obs in each gridbox
	ig=nint(xxg)
	jg=nint(yyg)
	if (ig.ge.1.and.ig.le.nx.and.jg.ge.1.and.jg.le.ny)
     .		obs(ig,jg)=obs(ig,jg)+1.

	cutoff=2.		 	  ! cutoff to limit search
	xcut=cutoff*xsc/dx		  ! look only twice the scale width
	ycut=cutoff*ysc/dy		  !   from the gridbox center

	do 100 i=1,nx			  ! loop on x gridpoints
	xgp=real(i)			  ! center of gridbox
	delx=abs(xgp-xxg)		  ! distance of data pt from grid ctr
	if (delx.gt.xcut .and. iwflag.eq.1) 
     .		delx=abs(delx-real(nx))   ! allow flagged time wrapping
	if (delx.gt.xcut) go to 100  	  ! only do nearby points

	do 101 j=1,ny			  ! loop on y gridpoints, same procedure
	ygp=real(j)		
	dely=abs(ygp-yyg)
	if (dely.gt.ycut) go to 101

	xgas=(delx*dx/xsc)**2		  	! make gaussian exponents
	ygas=(dely*dy/ysc)**2
	expn=exp(-xgas-ygas)		! make the gaussian weight
	wate(i,j)=wate(i,j)+expn	! sum the weights
	grid(i,j)=grid(i,j)+val*expn	! sum the weighted values

101	continue
100	continue

	return
	end

c************************************************************************
