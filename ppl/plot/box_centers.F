	SUBROUTINE BOX_CENTERS( x_or_y, val1, n, centers, valid )

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
* PLOT+ routine for use inside FERRET:
* Supply to PLOT+ the centers of n grid boxes from X or Y axis of the contour
* data currently in PLOT+.  Pointers to the appropriate FERRET axes are in
* COMMON.

* programmer - ACM
* (Adapted from box_bounds)
*
* for FERRET V62.00: 5/2009
* V62  Fixing bug 1659, plotting with subspan modulo axis.
* V73+ 1/18 *acm*  Changes for Issue 1009; precsision in time info sent to pplus for 2D plots
*             scaling for axis coords and box edges
* V766  4/22 *acm* For Issue 1009 the scaling of incoming val1 was incorrect

	include 'ferret.parm'
	include 'xbox_bounds.cmn'

* calling argument declarations:
	LOGICAL		valid
	INTEGER		n
	REAL		val1, centers(n), factr, offst
	CHARACTER	x_or_y*1

* internal variable declarations:
	INTEGER	ISUBSCRIPT, i, idim, isub1, isub0
	DOUBLE PRECISION TM_WORLD, val

* initialize
	IF ( x_or_y .EQ. 'X' ) THEN
	   idim = bb_xaxis
	   factr = bb_x_fac
	   offst = bb_x_off
	ELSEIF ( x_or_y .EQ. 'Y' ) THEN
	   idim = bb_yaxis
	   factr = bb_y_fac
	   offst = bb_y_off
	ENDIF

* determine starting subscript
	val1 = val1/factr - offst

	isub1 = ISUBSCRIPT ( DBLE(val1), bb_grid, idim, round_dn )
	isub0 = isub1 - 1

* pass back box centers

	DO i = 1, n
	   val1 = SNGL( TM_WORLD (isub0+i,bb_grid,idim,box_middle) )
	   centers(i) = (val1 + offst) * factr 
	ENDDO

* always (the appearance of) success
	valid = .TRUE.
	RETURN
	END
