	SUBROUTINE AXIS_INTERVALS(
     .				horiz_or_vert, mr, cx, idim, curvilinear,
     .				axis, lo_ss, hi_ss, lo_world, hi_world,
     .				n_regrid, d_regrid, regrid, irregular )

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
* convert required points of axis to single precision and determine
* re-gridding needs for non-constant spaced points

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/23/87
* revision 0.1 - 2/25/88 - added 2-direction Z-axis option
* revision 0.1 - 9/20/88 - create axis with MAKE_AXIS
* V200:  5/19/89 - reorder lo/hi_ss arguments
*	10/16/89 - eliminated invalid adjustable array subscripts
*        1/28/93 - allow axes of length 1
* V450: 3/97 - support for curvilinear coordinate plots
* V552: 5/03 - If the axis is known to be regular, dont do the test on 
*              whether "regrid" is needed. (this flagged fine axes as
*              irregular when in fact they arent).
* V553: 6/03 - the calculation of d_regrid for regular axes was wrong if 
*              theres a subspan-modulo void point
* v553: 7/03 - The initial d_regrid does need to start with 
*                axis( lo_cx + 1 ) - axis( lo_cx )
*              or it may miss an irregular part of the grid; the calculation
*              should be correct and not hit the subspan-modulo void point,
*              (how to test this???)

	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	LOGICAL	curvilinear, regrid, irregular
	INTEGER	mr, cx, idim, lo_ss, hi_ss, n_regrid
	REAL 	axis( lo_ss:hi_ss )
	REAL 	lo_world, hi_world, d_regrid
	CHARACTER*1 horiz_or_vert

* internal variable declarations:
	LOGICAL	BKWD_AXIS
	INTEGER	lo_cx, hi_cx, grid, i

* initialize
	grid   = mr_grid( mr )
	lo_cx = cx_lo_ss( cx, idim )
	hi_cx = cx_hi_ss( cx, idim )

* make a single precision axis corresponding to the subscript range of the
* memory variable (mr)
	IF ( curvilinear ) THEN
* ... axis of index values (other command line args provide true coordinates)
	   DO 100 i = lo_ss, hi_ss
 100	   axis(i) = i
	ELSE
	   CALL MAKE_AXIS( grid, idim, lo_ss, hi_ss, axis(lo_ss) )
	ENDIF

* single point?  Fake up some reasonable values.
        IF ( lo_ss .EQ. hi_ss ) THEN
           d_regrid = 1.
           IF ( axis(lo_cx) .EQ. 0.0 ) THEN
              lo_world = -1.
              hi_world =  1.
           ELSE
              lo_world = axis(lo_cx) * 0.99
              hi_world = axis(lo_cx) * 1.01
           ENDIF
           RETURN
        ENDIF

* determine if regridding is needed within the (cx) region to be plotted
* ... and determine smallest grid interval
	regrid = .FALSE.		! assume it's not needed
        d_regrid = axis( lo_cx + 1 ) - axis( lo_cx )

* The above calculation of d_regrid is wrong if theres a subspan-modulo void point

c        IF (lo_cx + 2 .LT. hi_cx) 
c     .      d_regrid = axis( lo_cx + 2 ) - axis( lo_cx + 1)

        IF (irregular) THEN
	   DO 200 i = lo_cx+1, hi_cx-1
* ... test within precision limits that may be visible in a plot
	      IF (ABS((axis(i+1)-axis(i))/d_regrid - 1.0) .GT. 3.E-6) THEN
		   regrid = .TRUE.
		   d_regrid = MIN( d_regrid, axis(i+1)-axis(i) )
	      ENDIF
 200	   CONTINUE
	ENDIF

* regridding parameters
	lo_world = axis(lo_cx)
	hi_world = axis(hi_cx)
	n_regrid = INT( (hi_world - lo_world) / d_regrid ) + 1

* plot vertical z axes upside down ?
	IF ( horiz_or_vert .EQ. 'Y' .AND. idim .EQ. z_dim ) THEN
	   IF ( BKWD_AXIS(idim, grid) ) THEN
	      lo_world = axis(hi_cx)
	      hi_world = axis(lo_cx)
	      d_regrid = -d_regrid
	      regrid   = .TRUE.
	   ENDIF
	ENDIF

	RETURN
      END
