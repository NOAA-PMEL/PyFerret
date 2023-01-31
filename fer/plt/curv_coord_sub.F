	SUBROUTINE CURV_COORD_SUB( xi, yj, npt, xfield, yfield,
     .				   xinv, yinv, status )

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
* convert the incoming coordinate arrays (xi,yj), which are "fractional
* indices" into true (X,Y) world coordinates by bi-linear interpolation
* in the coordinate arrays, xfield, yfield

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 
* V58+:  1/2005 ACM see if the longitude coordinates around x have
*                   crossed a branch cut, and if so correct the
*                   interpolants.
* V612: 7/08 *acm*  Fix bug 1589; if the units of the coordinates are not degrees
*                   then dont try to see if we are near the branch cut.
* V63   9/09 *acm* Fix bug 1688: curvlinear plots when the curv. coords are NOT
*                  longitude and latitude.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V687 *acm* 3/14 tickets 2151 and 2147: allow missing coordinate data in 3-arg 
*                 and 4-arg plot commands. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include	'xvariables.cmn'
	include	'xcurvilinear.cmn'

* calling argument declarations:
	INTEGER npt, status
	REAL*4  xinv, yinv, xlo, xhi
	REAL*8	xi(npt), yj(npt),
     .		xfield(xpos_loss_dim1:xpos_hiss_dim1,
     .		       xpos_loss_dim2:xpos_hiss_dim2),
     .		yfield(ypos_loss_dim1:ypos_hiss_dim1,
     .		       ypos_loss_dim2:ypos_hiss_dim2)

* internal variable declarations:
	INTEGER xlo_dim1, xhi_dim1, xlo_dim2, xhi_dim2,
     .		ylo_dim1, yhi_dim1, ylo_dim2, yhi_dim2, i
	REAL xfrac, yfrac, xit, yjt,
     .	     x11, x12, x21, x22,
     .	     y11, y12, y21, y22, aa
	CHARACTER*1 dim
        LOGICAL TM_FPEQ, lo, hi

        REAL branch_test, lon_degrees

	curv_missing = .FALSE.
        branch_test = 190.
        lon_degrees = 360.

	DO i = 1, npt

* transfer coordinate pair to temporary variables
	xit = xi(i)
	yjt = yj(i)

* clip to the limits of the axis subscripts
	IF ( .NOT.(dim1_modulo.OR.dim1_edges) ) THEN
	   IF (xit .LT. minss_dim1) xit = minss_dim1
	   IF (xit .GT. maxss_dim1) xit = maxss_dim1
	ENDIF
	IF ( .NOT.(dim2_modulo.OR.dim2_edges) ) THEN
	   IF (yjt .LT. minss_dim2) yjt = minss_dim2
	   IF (yjt .GT. maxss_dim2) yjt = maxss_dim2
	ENDIF

* determine the surrounding index values for both coordinates
* (only the X component for now ... Y conponent later)
	xlo_dim1 = INT(xit)
	IF ( FLOAT(xlo_dim1) .EQ. xit ) THEN
	   xhi_dim1 = xlo_dim1
	ELSE
	   IF (xit .LT. 0.) xlo_dim1 = xlo_dim1 - 1  ! neg val to round down
	   xhi_dim1 = xlo_dim1 + 1
	ENDIF
	xlo_dim2 = INT(yjt)
	IF ( FLOAT(xlo_dim2) .EQ. yjt ) THEN
	   xhi_dim2 = xlo_dim2
	ELSE
	   IF (yjt .LT. 0.) xlo_dim2 = xlo_dim2 - 1  ! neg val to round down
	   xhi_dim2 = xlo_dim2 + 1
	ENDIF

* determine the interpolation coefficients
	xfrac = xit - xlo_dim1
	yfrac = yjt - xlo_dim2

* apply modulo wrapping
* Note: this logic assumes that modulo wrapping is applied to indices
*   that are at most one point beyond the edge of the coordinate fields.
*   This is consistent with the index requirements of CONTOUR and SHADE
* e.g. an input of xi=0.5 would result in xlo_dim1=N, xhi_dim1=1
	IF ( dim1_modulo ) THEN
	   IF ( xlo_dim1 .LT. minss_dim1 ) xlo_dim1 = maxss_dim1
	   IF ( xlo_dim1 .GT. maxss_dim1 ) xlo_dim1 = minss_dim1
	   IF ( xhi_dim1 .LT. minss_dim1 ) xhi_dim1 = maxss_dim1
	   IF ( xhi_dim1 .GT. maxss_dim1 ) xhi_dim1 = minss_dim1
	ENDIF
	IF ( dim2_modulo ) THEN
	   IF ( xlo_dim2 .LT. minss_dim2 ) xlo_dim2 = maxss_dim2
	   IF ( xlo_dim2 .GT. maxss_dim2 ) xlo_dim2 = minss_dim2
	   IF ( xhi_dim2 .LT. minss_dim2 ) xhi_dim2 = maxss_dim2
	   IF ( xhi_dim2 .GT. maxss_dim2 ) xhi_dim2 = minss_dim2
	ENDIF

* Propagate the X indices to the Y field (to be modified next)
	ylo_dim1 = xlo_dim1
	yhi_dim1 = xhi_dim1
	ylo_dim2 = xlo_dim2
	yhi_dim2 = xhi_dim2

* Set up for degenerate coordinate fields (single point on an axis)
* e.g.  SHADE vals2D xpos1d ypos1d
	IF (xspan_1_dim1) THEN
	   xlo_dim1 = minss_dim1
	   xhi_dim1 = xlo_dim1
	ENDIF
	IF (xspan_1_dim2) THEN
	   xlo_dim2 = minss_dim2
	   xhi_dim2 = xlo_dim2
	ENDIF
	IF (yspan_1_dim1) THEN
	   ylo_dim1 = minss_dim1
	   yhi_dim1 = ylo_dim1
	ENDIF
	IF (yspan_1_dim2) THEN
	   ylo_dim2 = minss_dim2
	   yhi_dim2 = ylo_dim2
	ENDIF

* The indexing of the X and Y fields may each be offset from the indices given
	xlo_dim1 = xlo_dim1 + xpos_off0_dim1
	xhi_dim1 = xhi_dim1 + xpos_off0_dim1
	xlo_dim2 = xlo_dim2 + xpos_off0_dim2
	xhi_dim2 = xhi_dim2 + xpos_off0_dim2

	ylo_dim1 = ylo_dim1 + ypos_off0_dim1
	yhi_dim1 = yhi_dim1 + ypos_off0_dim1
	ylo_dim2 = ylo_dim2 + ypos_off0_dim2
	yhi_dim2 = yhi_dim2 + ypos_off0_dim2

* get the interpolants
*	        (x,y)22           (x,y)22
*	
*	             	 (ix,jy)
*	
*	        (x,y)11           (x,y)12

	x11 = xfield(xlo_dim1, xlo_dim2)
	x12 = xfield(xhi_dim1, xlo_dim2)
	x21 = xfield(xlo_dim1, xhi_dim2)
	x22 = xfield(xhi_dim1, xhi_dim2)

	y11 = yfield(ylo_dim1, ylo_dim2)
	y12 = yfield(yhi_dim1, ylo_dim2)
	y21 = yfield(ylo_dim1, yhi_dim2)
	y22 = yfield(yhi_dim1, yhi_dim2)

* check that all of the needed interpolants exist3/24/2014
	IF ( TM_FPEQ(x11,badxcurv)
     .	.OR. TM_FPEQ(x12,badxcurv)
     .	.OR. TM_FPEQ(x21,badxcurv)
     .	.OR. TM_FPEQ(x22,badxcurv)) THEN
	   dim = 'X'
	   GOTO 5100
	ELSEIF (
     .	     TM_FPEQ(y11,badycurv)
     .	.OR. TM_FPEQ(y12,badycurv)
     .	.OR. TM_FPEQ(y21,badycurv)
     .	.OR. TM_FPEQ(y22,badycurv) ) THEN
	   dim = 'Y'
	   GOTO 5100
	ENDIF

C Check to see if any of the longitudes have crossed a branch cut.
C add or subtract local coordinates to correct for this. Puts all 
C coordinates on the same branch as x11

       IF (units_degrees) THEN

          IF (ABS(x11-x12) .GT. branch_test) THEN
              IF (x11 .GT. x12) THEN
                 x12 = x12 + lon_degrees
              ELSE
                 x12 = x12 - lon_degrees
              ENDIF
           ENDIF
           IF (ABS(x11-x21) .GT. branch_test) THEN
              IF (x11 .GT. x21) THEN
                 x21 = x21 + lon_degrees
              ELSE
                 x21 = x21 - lon_degrees
              ENDIF
           ENDIF
           IF (ABS(x11-x22) .GT. branch_test)   THEN
              IF (x11 .GT. x22) THEN
                 x22 = x22 + lon_degrees
              ELSE
                 x22 = x22 - lon_degrees
              ENDIF
           ENDIF

        ENDIF ! units_degrees

* perform the interpolation
	xi(i) = (x11*(1.-xfrac) + x12*(xfrac)) * (1.-yfrac)
     .	      + (x21*(1.-xfrac) + x22*(xfrac)) * (yfrac)
	yj(i) = (y11*(1.-xfrac) + y12*(xfrac)) * (1.-yfrac)
     .	      + (y21*(1.-xfrac) + y22*(xfrac)) * (yfrac)

	ENDDO ! i = 1, npt

* Check longitudes for a branch point

        IF (units_degrees) THEN
           x11 = xi(1)
           DO i = 2, npt
              aa = ABS(x11-xi(i))
              IF (ABS(x11-xi(i)) .GT. branch_test) THEN
                 IF (x11 .GT. xi(i)) THEN
                    xi(i) = xi(i) + lon_degrees
                 ELSE
                    xi(i) = xi(i) - lon_degrees
                 ENDIF
              ENDIF
	   ENDDO
        ENDIF ! units_degrees

* possibly invert the sign of the coordinates (used for descending Z axis)
	IF ( xinv .EQ. -1. ) THEN
	   DO i = 1, npt
	      xi(i) = -1.*xi(i)
 	   ENDDO
	ENDIF
	IF ( yinv .EQ. -1. ) THEN
	   DO i = 1, npt
	      yj(i) = -1.*yj(i)
 	   ENDDO
	ENDIF

* successful completion
	status = 0  ! (like ferr_ok but for PLOT+)
	RETURN

* error exit
 5000	RETURN
 5100	CONTINUE
	curv_missing = .TRUE.
	GOTO 5000
c	CALL ERRMSG( ferr_out_of_range, status,
c     .		dim//' coordinates'//pCR//
c     .		'missing value where val is needed', *5000)

	END 
