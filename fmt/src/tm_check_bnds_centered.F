	LOGICAL FUNCTION TM_CHECK_BNDS_CENTERED (line, line_edges, npts, 
     .		iaxis, doubletype)
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
* Checks whether data points are all centered within the bounds on an axis
* Do this using the same precision as the coordinates, as in cd_get_1_axis.
*
* Returns FALSE for any irregular axis, even if the coordinates are centered
* in the irregular grid cells. This is for detecting an axis whose coordinates
* are regularly spaced but the grid cells are off-center, in which case the
* coordinates and bounds must be saved.

* 4/2004 ACM 
* V58+ *acm*  1/05 Fixes for irreg axes with bounds: This routine now checks
*                  whether the coordinates are centered in the bound boxes, and 
*                  whether the boxes are equal sized.
* V601 *acm* 9/19  Fix bug 1434; check bounds for regularly-spaced axes 
* V694  5/15 *acm* Match precision for bounds checks (as is done for coords).
* V72 6/17 *acm* for trac enhancement #767 -- dynamic coordinate storage 
*                line and line_edges are the coordinates and edges for this line
* V745 12/18 *acm* Noticed that regular clim axes from clim_axes_gregorian.nc were 
*                  shown as irregular ( MON_REG_GRG TIME 12mi). Use epsilon test 
*                  for comparing  diff_hi and diff_last (!)
* V65  2/23 *acm* Add a sanity-check on irregular axis check. The code checked
*		  that the half-grid-box size is the same as the one from the 
*		  previous cell.  Also check if the grid box is the same size as
*		  the first grid box. A long axis with tiny variations could
*		  appear regular with just the first check.

	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* Argument declarations
	REAL	line(*), line_edges(*)
	LOGICAL doubletype, regular
	INTEGER npts, iaxis

* Local variable declarations
        LOGICAL TM_FPEQ_EPS, TM_DFPEQ_TOL, equal
	INTEGER i
        REAL diff_hi, diff_lo, diff_last
        REAL pt, del, blo, bhi, box_first
	REAL epsilon_21, epsilon
	REAL*8 depsilon_42, depsilon, first_delta

	epsilon_21 =  2.**(-21) 
	depsilon_42 =  2.**(-42) 

	first_delta = line(2) - line(1)

	depsilon = depsilon_42 * 2.*(ABS(line(1)) / first_delta )
	epsilon = epsilon_21 * 2.*(ABS(line(1)) / first_delta )

        IF (line_regular(iaxis)) THEN
           pt = line_start(iaxis)
           del = line_delta(iaxis) 

	   DO i = 1, npts

             blo = line_edges(2*(i-1)+1)
             bhi = line_edges(2*(i-1)+2)

	     diff_hi = bhi - pt
             diff_lo = pt - blo

	     IF (doubletype) THEN
	       equal = TM_DFPEQ_TOL(diff_lo, diff_hi, depsilon)
	     ELSE
	       equal = TM_FPEQ_EPS(epsilon, diff_lo, diff_hi) 
	     ENDIF

	     IF ( .NOT. equal ) GOTO 500


             IF (i .GT. 1) THEN
                
		IF (doubletype) THEN
	           equal = TM_DFPEQ_TOL(diff_hi, diff_last, depsilon) 
		ELSE
	           equal = TM_FPEQ_EPS(epsilon, diff_hi, diff_last) 
		ENDIF
		IF ( .NOT. equal ) GOTO 500
		
	     ENDIF
             diff_last = diff_hi
             pt = float(i) * del + line_start(iaxis)

	   ENDDO

        ELSE  ! not line_regular
   
	   box_first = line_edges(2) - line_edges(1)
	   DO  i = 1, npts
             blo = line_edges(2*(i-1)+1)
             bhi = line_edges(2*(i-1)+2)

             diff_hi = bhi - line(i)
             diff_lo = line(i)- blo

* Check for centered coordinate 
	     IF (doubletype) THEN
	       equal = TM_DFPEQ_TOL(diff_lo, diff_hi, depsilon)
	     ELSE
	       equal = TM_FPEQ_EPS(epsilon, diff_lo, diff_hi) 
	     ENDIF
	     IF ( .NOT. equal ) GOTO 500

* Further check on equal box sizes: Is the half grid-box size the same as 
* the previous cell, and also is the box size is the same as the first box

             IF (i .GT. 1) THEN

		IF (doubletype) THEN
	           equal = TM_DFPEQ_TOL(diff_hi, diff_last, depsilon) 
		ELSE
	           equal = TM_FPEQ_EPS(epsilon, diff_hi, diff_last) 
		ENDIF
	        IF ( .NOT. equal ) GOTO 500
		
		IF (doubletype) THEN
	           equal = TM_DFPEQ_TOL(bhi-blo, box_first, depsilon) 
		ELSE
	           equal = TM_FPEQ_EPS(epsilon, bhi-blo, box_first) 
		ENDIF
	        IF ( .NOT. equal ) GOTO 500

	     ENDIF
             diff_last = diff_hi

	   ENDDO

        ENDIF

	TM_CHECK_BNDS_CENTERED = .TRUE.
	RETURN

 500	CONTINUE
	TM_CHECK_BNDS_CENTERED = .FALSE.
	RETURN

	END
