      SUBROUTINE RE_ORIENT( xlo, xhi, ylo, yhi,
     .			    dim1, dim2, md1, md2, transpz, aspect,
     .			    ctx, xgrid, mx, ygrid, my, sout )

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
*
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
* V600  4/06 *acm* - make refinement to cos(lat) factor as suggested by A. Wittenberg
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* 6/06 - Fix to Andrew*s correction for XY integ., bug1348
*12/06 - Fix to cos_factor calculation was STILL wrong. Try VEC/ASPECT plots 
*        to see the error.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V740: 2/18 *acm* for issue 1860, apply a cos(lat) correction
*                  in vector planes such as XZ

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

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include	'xvariables.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	LOGICAL	transpz
	INTEGER	dim1, dim2, md1, md2, mx, my, ctx
	REAL	aspect, sout
	REAL	xgrid( md1, md2 ), ygrid( md1, md2 )
	REAL*8	xlo, xhi, ylo, yhi

* internal variable declarations:
        LOGICAL GEOG_COS_FACTOR, TM_FPEQ, need_cos, fixed_cos
	INTEGER	grid, i, j, j0
	REAL	AXIS_DISTANCE, BOX_SIZE, TM_WORLD, 
     .          s0, s, a, b, tan, tansq, ssq, dx, dy,bad_x, bad_y, 
     .          cos_factor, bx, yh_2, yc, ybxlo, ybxhi, yloc, ybx

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
* Check whether there is an x axis on the plot and a latitude axis orthogonal to 
* the plane being plotted - fixed cosine correction.

	need_cos =  .FALSE.
	fixed_cos = .FALSE.

* For vectors in a plane involving X but not Y, there may need to be
* a cos(lat) correction

	yloc =  unspecified_val8
	IF ( (dim1.EQ.x_dim .OR. dim2.EQ.x_dim) .AND. 
     .       (dim1.NE.y_dim .AND. dim2.NE.y_dim) ) THEN
	   yloc = cx_lo_ww(y_dim,ctx)
	   IF (yloc .EQ. unspecified_val8 ) yloc = 
     .         TM_WORLD( cx_lo_ss(ctx,y_dim), grid, y_dim, box_middle )
	   ybx = BOX_SIZE(cx_lo_ss(ctx,y_dim), grid, y_dim )
	   fixed_cos = GEOG_COS_FACTOR(x_dim, grid)
     .              .AND. GEOG_COS_FACTOR(y_dim, grid)
	   yc = deg2rad * yloc
	   yh_2 = 0.5*deg2rad * ybx
	   cos_factor =  COS( yc ) * ABS( SIN(yh_2) )/yh_2
	ENDIF

	IF ( dim1.EQ.x_dim .AND. dim2.EQ.y_dim ) THEN
           need_cos =     GEOG_COS_FACTOR(x_dim, grid)
     .              .AND. GEOG_COS_FACTOR(y_dim, grid)
       ENDIF

* re-orient each point
	DO 110 j = 1, md2

* compute COS(lat) correction at this latitude
* 3/06 - apply Andrew*s suggested correction, bug1348

	   IF ( need_cos ) THEN

c Previous calculation
              yc = deg2rad * (TM_WORLD(j+j0,grid,y_dim,box_middle))
              cos_factor = COS( yc )
  
c Correction
              bx = BOX_SIZE(j+j0, grid, y_dim )
	      yh_2 = 0.5*deg2rad * bx
              cos_factor =  COS( yc ) * ABS( SIN(yh_2) )/yh_2

C The above are all OK if the coordinate point is in the middle of the box.
C If the coordinate is not in the middle of the cell then it should be
C ABS(SIN(yhi)-SIN(ylo))  
C See this thread:
C http://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2009/msg00506.html
C And also ticket 1348.

              ybxlo = deg2rad* 
     .          ( TM_WORLD(j+j0,grid,y_dim,box_lo_lim) )
              ybxhi = deg2rad* 
     .          ( TM_WORLD(j+j0,grid,y_dim,box_hi_lim) )

              cos_factor = ABS( SIN(ybxhi) - SIN(ybxlo) )/(deg2rad * bx)

              IF ( cos_factor .LT. 1.E-6 ) cos_factor = 1.E-6 !arbitrary fix /0
           ELSE
              IF (.NOT.fixed_cos) cos_factor = 1.0
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
