c
c	ppl
C** 
C**    @(#)PPL.INC	1.2    9/8/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
	common/ppl/ppts,pzmin,pzmax,pzmean,pxmin,pxmax,pnx,
     *	pymin,pymax,pny,ptype,pxlab(20),pylab(20),pmlab(30),
     *	pdate(4),ptime(4),pident(8),psrc(8),
     *	varnme(8),vmean(4),vvar(4),vstd(4),vrms(4),vmin(4),vmax(4),
     *	vdate(4),vtime(4),vjdate(4),vident(8),vlat(4),vlong(4),
     *	vxmean,vxvar,vxmin,vxmax,vxjdat,vxlat,vxlong,
     *	gtype(2),gornt(2),grid(8),
     *	gxornt,
     *	cmnta(30),cmntb(30),cmntc(30),cmntd(30),cmnte(30),
     *	usera(8),userb(8),userc(8),userd(8)
c
	real	  ppts,pzmin,pzmax,pzmean,pxmin,pxmax,pymin,pymax
	integer*4 pnx,pny,ptype
	integer*4 pxlab,pylab,pmlab,pdate,ptime,pident,psrc
	integer*4 varnme,vmean,vvar,vstd,vrms,vmin,vmax
	integer*4 vdate,vtime,vjdate,vident,vlat,vlong
	real	  vxmean,vxvar,vxmin,vxmax,vxjdat,vxlat,vxlong
	integer*4 gtype,gornt,grid
	real	  gxornt
	integer*4 cmnta,cmntb,cmntc,cmntd,cmnte
	integer*4 usera,userb,userc,userd
c
c	DSF header for FORMAT= PPLUS files.  (note: not BIBO files)
c
c	note:  all labels are integer hollerith!!
c
c	ppts	actual number of points in data file (y and x) (real)
c	pzmin	minimum value for x and y  (real)
c	pzmax	maximum value for x and y  (real)
c	pzmean	mean for x and y  (real)
c	pxmin	minimum x value  (real) (user units)
c	pxmax	maximum x value  (real) (user units)
c	pnx	number of x grid points (integer)
c	pymin	minimum y value (real) (user units)
c	pymax	maximum y value (real) (user units)
c	pny	number of y grid points (integer)
c	ptype	=2 2-d irregular rectangular grid
c		=1 1-d set
c		=0 2-d regular rectangular grid (integer)
c		=-1 2-d second component of a vector regular grid
c		=-2 2-d second component of a vector irregular grid
c	pxlab	xaxis label 80
c	pylab	y axis label 80
c	pmlab	main title of plot 120
c
c	pdate	date of plot file creation 16
c	ptime	time of plot file creation 16
c	pident	identifier of plot file 32
c	psrc	source of plot file 32
c
c	varnme	name of variable 32
c	vmean	mean of var 16
c	vvar	variance of variable 16
c	vstd	standard deviation of variable 16
c	vrms	root mean square of variable 16
c	vmin	minimum of variable 16
c	vmax	maximum of variable 16
c	vdate	gregorian date 16
c	vtime	time 16
c	vjdate	julian date of var 16
c	vident	variable identification 32
c	vlat	latitude of variable (or grid) 16
c	vlong	longitude of variable (or grid) 16
c	vxmean	mean of variable (real)
c	vxvar	variance of variable (real)
c	vxmin	minimum of variable (real)
c	vxmax	maximum of variable (real)
c	vxjdat	julian date of var (real, modified)
c	vxlat	latitude of variable (real)
c	vxlong	longitude of variabel (real)
c
c	gtype	grid type 8
c	gornt	orientation of grid 8
c	grid	additional grid information 32
c	gxornt	orientation of grid (real)
c
c	cmnta	comment a 120
c	cmntb	comment b 120
c	cmntc	comment c 120
c	cmntd	comment d 120
c	cmnte	comment e 120
c	usera	user field a 32
c	userb	user field b 32
c	userc	user field c 32
c	userd	user field d 32
c
c*******************************************************************
