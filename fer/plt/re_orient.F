      SUBROUTINE RE_ORIENT( xlo, xhi, ylo, yhi,
     .			    dim1, dim2, md1, md2, transpz, aspect,
     .			    xgrid, mx, ygrid, my, sout )

* re-orient vectors so that they point in the correct direction on the plot
* preserving the overall magnitude

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 4/1/88
* V200:  5/19/89 - 4D symmetrical
* V230:  1/24/92 - incorporate COS(lat) if appropriate
* V301:  11/93 - replace exact zero check with FPEQ
* V420	 2/96 - calling args changed so /XLIMITS and /YLIMITS can be passed
*	      - also sout, the returned scale factor, was wrong on plots
*		that used the COS factor - was based on "s" instead of "s0"

* procedure:	at each point
*	   x --> xprime = A * x
*	   y --> yprime = B * y
*	where
*	   x**2 + y**2 = xprime**2 + yprime**2
*	and  A/B = s, a constant

*	let dx		the range of data on the x axis
*	    dy		the range of data on the y axis
*	    aspect 	the ratio of axis lengths ylen/xlen
*	    xscale	the scale factor (units/inch) on the x axis
*	    s		the ratio of scale factors, yscale/xscale
*	    tan		the ratio of the vector components y/x at any point
*	then
*	    s = (dy/dx)/aspect

*	and, from the constraints on the transformation, above
*	    B = { (1 + tan**2) / (s**2 + tan**2) } **.5
*	    A = s * B

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	LOGICAL	transpz
	INTEGER	dim1, dim2, md1, md2, mx, my
	REAL	aspect, sout
	REAL	xgrid( md1, md2 ), ygrid( md1, md2 )
	REAL*8	xlo, xhi, ylo, yhi

* internal variable declarations:
        LOGICAL need_cos, GEOG_COS_FACTOR, TM_FPEQ
	INTEGER	grid, i, j, j0
	REAL	AXIS_DISTANCE, s0, s, a, b, tan, tansq, ssq, dx, dy,
     .          bad_x, bad_y, cos_factor
	REAL*8	TM_WORLD

* initialize
	grid  = mr_grid( mx )
	bad_x = mr_bad_data( mx )
	bad_y = mr_bad_data( my )
        j0 = mr_lo_ss( mx, y_dim ) - 1  ! may need for latitude cuz j=1,md2

* axis spans in km
	dx = AXIS_DISTANCE( xlo, xhi, dim1, grid )
	dy = AXIS_DISTANCE( ylo, yhi, dim2, grid )
	s0   = (dy/dx) / aspect   ! axis scale ratio before COS(lat) correction

* will we need COS(latitude) corrections ? ( the true dx may be dx*COS(lat) )
        IF ( dim1.EQ.x_dim .AND. dim2.EQ.y_dim ) THEN
           need_cos =     GEOG_COS_FACTOR(x_dim, grid)
     .              .AND. GEOG_COS_FACTOR(y_dim, grid)
        ELSE
           need_cos = .FALSE.
        ENDIF

* re-orient each point
	DO 110 j = 1, md2

* compute COS(lat) correction at this latitude
           IF ( need_cos ) THEN
              cos_factor = COS( deg2rad *
     .                     SNGL( TM_WORLD(j+j0,grid,y_dim,box_middle)) )
              IF ( cos_factor .LT. 1.E-6 ) cos_factor = 1.E-6 !arbitrary fix /0
           ELSE
              cos_factor = 1.0
           ENDIF
           s = s0 / cos_factor
           IF ( transpz ) s = 1. / s
           ssq  = s*s

	   DO 100 i = 1, md1

* skip flagged points
	      IF ( xgrid(i,j).EQ.bad_x .OR. ygrid(i,j).EQ.bad_y ) THEN
	         xgrid(i,j) = bad_x
	         ygrid(i,j) = bad_y
	         GOTO 100
	      ENDIF

* no rotation on vertical (or horizontal) vectors (avoid y/0)
	      IF ( TM_FPEQ(xgrid(i,j), 0.0) ) GOTO 100

	      tansq = ( ygrid(i,j)/xgrid(i,j) ) ** 2
	      b     = SQRT( (1.+tansq) / (ssq+tansq) )
	      a     = b * s

	      xgrid(i,j) = a * xgrid(i,j)
	      ygrid(i,j) = b * ygrid(i,j)

 100	   CONTINUE
 110    CONTINUE

* return 1/s to display on plot (generally, s<1)
	sout = 1. / s0			! bug fix 2/96: was 1./s 

	RETURN
	END
