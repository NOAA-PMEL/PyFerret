	SUBROUTINE GET_COEF_FOR_XACT( src_lo, src_hi, src_grid,
     .				    dst_lo, dst_hi, dst_grid,
     .				    axis, ss, coef )

*get_linear_coef
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* for regridding axis axis from source grid to destination grid:
* determine linear interpolation coefficients and nearest lower neighbor point 

* programmer - Ansley Manke, based on get_linear_coef.  For XACT regridding
* need to determine an appropriate precision for the coefficient matching, to.

        IMPLICIT NONE
	include 'ferret.parm'
	include 'errmsg.parm'

* calling argument declarations:
	INTEGER src_lo, src_hi, dst_lo, dst_hi, src_grid, dst_grid, axis
	INTEGER ss(dst_lo:dst_hi)
	REAL	coef (dst_lo:dst_hi)

* internal variable declarations:
	LOGICAL TM_FPEQ_SNGL, TM_DFPEQ_TOL, eqflag
	INTEGER	ISUBSCRIPT, isrc, ii, status
	REAL*8	TM_WORLD, TDEST_WORLD, xsrc, xdst, xmin, xmax, 
     .		xlast, dhi, dlo, tol, tmag, count
	REAL*4  rsrc, rdst

* initialize
	IF ( axis .EQ. t_dim .OR. axis .EQ. f_dim ) THEN
*          TDST_WORLD gives dates from the source axis as they would be
*	   encoded on the destination axis 
*      ==> ALL TIME/DATE CALCULATIONS ARE DONE IN THE SOURCE AXIS ENCODING
	   CALL TDEST_WORLD_INIT ( dst_grid, src_grid, axis, status ) ! src<->dst
	   IF ( status .NE. ferr_ok ) RETURN  ! note - status not passed !!
	ENDIF
	xmin = TM_WORLD( src_lo, src_grid, axis, box_middle )
	xmax = TM_WORLD( src_hi, src_grid, axis, box_middle )

* loop through points of destination axis to compute a tolerance
	xlast = TDEST_WORLD( dst_lo, dst_grid, axis, box_middle )
	dhi = 0.
	dlo = TDEST_WORLD( dst_hi, dst_grid, axis, box_middle ) - xlast
	tmag = xlast

	DO ii = dst_lo+1, dst_hi
	   xdst = TDEST_WORLD( ii, dst_grid, axis, box_middle )
	   dhi = MAX(dhi, xdst-xlast)
	   dlo = MIN(dlo, xdst-xlast)
	   tmag = tmag + xdst
	   xlast = xdst
	ENDDO

* The tolerance, when deltas are small, computed from the axis precision
* and the magnitude of the coords. 
	tol = (dhi+dlo)/2  ! lo and hi delta (should this be some other statistic?)
	tmag = tmag/(dst_hi-dst_lo+1) ! average coordinate magnitude

	tol = tol/tmag
	IF (LOG10(tol) .LT. -2.) THEN
	   tmag = INT(LOG10(tol)-1)
	   tol = 10**tmag
	ENDIF

* loop through points of destination axis
	DO 100 ii = dst_lo, dst_hi

* get destination grid point in question
* if T axis, then get it encoded as per source time axis
	   xdst = TDEST_WORLD( ii, dst_grid, axis, box_middle )
	   rdst = xdst

	   IF ( SNGL(xdst) .LT. SNGL(xmin)
     .     .OR. SNGL(xdst) .GT. SNGL(xmax) ) THEN  	! SNGL 5/93

* ... destination point does not lie within source grid bounds
	      coef( ii ) = bad_val4
	      ss  ( ii ) = unspecified_int4

	   ELSE

	      isrc = ISUBSCRIPT( xdst, src_grid, axis, round_dn )
	      xsrc = TM_WORLD( isrc, src_grid, axis, box_middle )
	      rsrc = xsrc

	      eqflag = TM_FPEQ_SNGL(rdst,rsrc) 
	      IF (tmag.LT.0) eqflag = TM_DFPEQ_TOL(xdst,xsrc,tol) 

	      IF ( eqflag ) THEN   
	         ss  ( ii ) = isrc
	         coef( ii ) = 0.0
	      ELSEIF ( xdst .GT. xsrc ) THEN
	         ss  ( ii ) = isrc
	         coef( ii ) =	( xdst - xsrc ) /
     .			( TM_WORLD(isrc+1,src_grid,axis,box_middle) - xsrc )
	      ELSE
	         ss  ( ii ) = isrc - 1
	         coef( ii ) =	1.0 - ( ( xsrc - xdst ) /
     .			( xsrc - TM_WORLD(isrc-1,src_grid,axis,box_middle) ) )
	      ENDIF

	   ENDIF

 100	CONTINUE

	RETURN
	END
