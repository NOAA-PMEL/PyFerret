	SUBROUTINE GET_LINEAR_COEF( src_lo, src_hi, src_grid,
     .				    dst_lo, dst_hi, dst_grid,
     .				    axis, ss, coef )

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
* for regridding axis axis from source grid to destination grid:
* determine linear interpolation coefficients and nearest lower neighbor point 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 12/30/88
* revision 0.1 -  1/15/89 - fixed bug in coef for xdst<xsrc
* V200:  8/7/89 - allow exact equality case
* V230:  8/23/92 - bug fix for non-date time axes
* V300:   5/5/93 - coordinate comparisons in single precision
* V312:	  6/7/94 - use TDST_WORLD* for simpler, faster date conversions

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations:
	INTEGER src_lo, src_hi, dst_lo, dst_hi, src_grid, dst_grid, axis
	INTEGER ss(dst_lo:dst_hi)
	REAL	coef (dst_lo:dst_hi)

* internal variable declarations:
	INTEGER	ISUBSCRIPT, isrc, ii, status
	REAL*8	TM_WORLD, TDEST_WORLD, xsrc, xdst, xmin, xmax

* initialize
	IF ( axis .EQ. t_dim ) THEN
*          TDST_WORLD gives dates from the source axis as they would be
*	   encoded on the destination axis 
*      ==> ALL TIME/DATE CALCULATIONS ARE DONE IN THE SOURCE AXIS ENCODING
	   CALL TDEST_WORLD_INIT ( dst_grid, src_grid, status ) ! src<->dst
	   IF ( status .NE. ferr_ok ) RETURN  ! note - status not passed !!
	ENDIF
	xmin = TM_WORLD( src_lo, src_grid, axis, box_middle )
	xmax = TM_WORLD( src_hi, src_grid, axis, box_middle )

* loop through points of destination axis
	DO 100 ii = dst_lo, dst_hi

* get destination grid point in question
* if T axis, then get it encoded as per source time axis
	   xdst = TDEST_WORLD( ii, dst_grid, axis, box_middle )

	   IF ( SNGL(xdst) .LT. SNGL(xmin)
     .     .OR. SNGL(xdst) .GT. SNGL(xmax) ) THEN  	! SNGL 5/93

* ... destination point doesn't lie within source grid bounds
	      coef( ii ) = bad_val4
	      ss  ( ii ) = unspecified_int4

	   ELSE

	      isrc = ISUBSCRIPT( xdst, src_grid, axis, round_dn )
	      xsrc = TM_WORLD( isrc, src_grid, axis, box_middle )
	      IF ( SNGL(xdst) .EQ. SNGL(xsrc) ) THEN    ! SNGL 5/93
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
