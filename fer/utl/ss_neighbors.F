	SUBROUTINE SS_NEIGHBORS ( world_coord, grid, idim,
     .				  extrapolate, sslo, sshi, factor )

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
* coordinate on the indicated axis.
* Return also the linear interpolation factor with respect to the first
* subscript.
* When the value is out of range unspecified_int4 is returned.
* If the value is before the beginning or after the end of the axis but
* still inside the limits of the end grid box then both high and low
* subscripts will be set to this subscript and "extrapolate" will
* be set to true indicating that this is a questionable value.
* The values sslo and sshi are both input and output values.  On input they
* are used to optimize the search on the assumption that the next point
* is likely to be close to the last.
* If the indicated axis is modulo then the subscript value on the "root"
* axis is returned.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V301 - 2/94 - based on ISUBSCRIPT
* V420 - 10/95 - allow line defs to be recursive (dynamic lines) by avoiding
*		 direct use of line_mem

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	LOGICAL extrapolate
	INTEGER	grid, idim, sslo, sshi
	REAL	world_coord, factor

* internal variable declarations:
	INTEGER	lo_look, hi_look, iaxis, ibot, itop, look,
     .		isub0, line_len, isubsc, num_mod, point
	REAL 	subsc
	REAL*8	TM_WORLD, wcoord, start, aend, span, first, last, wwindex

* initialize 
	wcoord = world_coord	! switch to REAL*8
	extrapolate = .FALSE.

* which axis are we on ?
	iaxis    = grid_line ( idim, grid )
	line_len = line_dim( iaxis )

* check for peculiar calls
	IF (  iaxis .EQ. mnormal
     .	 .OR. iaxis .EQ. munknown ) THEN
	   sslo = unspecified_int4
	   RETURN
	ENDIF

* initialize pointers for axes with irregularly spaced points
	IF ( .NOT.line_regular( iaxis ) ) THEN
	   isub0 = line_subsc1( iaxis ) - 1
	   ibot  = isub0 + 1
	   itop  = isub0 + line_len
	ENDIF

* get positions of first and last points on axis
	IF ( line_regular( iaxis ) ) THEN
	   first = line_start( iaxis )
	   last  = first + line_delta( iaxis ) * (line_len-1)
        ELSE
           first = TM_WORLD(       1, grid, idim, box_middle)
           last  = TM_WORLD(line_len, grid, idim, box_middle)
	ENDIF

* ***** SOLUTIONS WHEN GIVEN POINT BEYOND AXIS ENDS
	IF ( wcoord .LT. first .OR. wcoord .GT. last ) THEN
* ... begin by computing axis limits as outer edges of end boxes
	   IF ( line_regular( iaxis ) ) THEN
	      span  = line_delta( iaxis ) * line_len
	      start = line_start( iaxis ) - line_delta( iaxis )/2.
	      aend  = start + span
           ELSE
              start = TM_WORLD(       1, grid, idim, box_lo_lim)
              aend  = TM_WORLD(line_len, grid, idim, box_hi_lim)
              span  = aend - start
	   ENDIF

* Modulo axis: translate world coord to within axis range
	   IF ( line_modulo( iaxis ) ) THEN
* ... now translate to within axis limits saving modulo number
	      IF     ( wcoord .GT. aend  ) THEN
	         num_mod = ( wcoord - start ) / span
! 10/92 bug fix: if wcoord is an exact multiple of length (e.g. wccord=24.
!       start=0., span=12.) then modulo remap to top of ax rather than bottom
                 IF ( wcoord .EQ. start+num_mod*span )
     .                          num_mod = num_mod - 1
	      ELSEIF ( wcoord .LT. start ) THEN
	         num_mod = ( wcoord - aend  ) / span		! (negative)
	      ELSE
	         num_mod = 0
	      ENDIF
	      wcoord = wcoord - num_mod*span
	   ENDIF

* ... three cases: modulo (always solution), exrapolate(maybe sol'n), or
*     not extrap. (nexer solution)
	   IF ( wcoord .LT. first ) THEN
	      IF ( line_modulo(iaxis) ) THEN
	         sslo = 1
	         sshi = line_len
	         factor = 1. - (first-wcoord)/(first+span-last)
	         RETURN
	      ELSEIF ( wcoord .GE. start ) THEN
	         point = 1
	         extrapolate = .TRUE.
	         GOTO 1100
	      ELSE
	         GOTO 1000		! no solution
	      ENDIF
	   ELSEIF ( wcoord .GT. last ) THEN
	      IF ( line_modulo(iaxis) ) THEN
	         sslo = 1
	         sshi = line_len
	         factor = (wcoord-last)/(first+span-last)
	         RETURN

	      ELSEIF ( wcoord .LE. aend ) THEN
	         point = line_len
	         extrapolate = .TRUE.
	         GOTO 1100
	      ELSE
	         GOTO 1000		! no solution
	      ENDIF
	   ELSEIF ( wcoord .EQ. first ) THEN	! possibly unnecessary test
	      point = 1
	      GOTO 1100		! exact point   
	   ELSEIF ( wcoord .EQ. last ) THEN	! possibly unnecessary test
	      point = line_len
	      GOTO 1100
	   ENDIF
	ENDIF
* ***** END OF SOLUTIONS BEYOND AXIS ENDS

* ***** AXIS POINTS ARE REGULARLY SPACED --> QUICK CALCULATION!
	IF ( line_regular( iaxis ) ) THEN
	   subsc = (wcoord-first)/line_delta(iaxis)
	   isubsc = INT( subsc )
	   IF ( subsc .EQ. FLOAT(isubsc) ) THEN
	      point = isubsc + 1
	      GOTO 1100
	   ELSE
	      sslo = isubsc + 1
	      sshi = isubsc + 2
	      factor = 1. - (subsc - FLOAT(isubsc))	      
	   ENDIF
	   RETURN
* ***** END OF REGULARLY SPACED SOLUTIONS

	ELSE			! AXIS POINTS ARE IRREGULAR !!!!

! This optimization is untested and was never compiled because searches
! of irregluar axes are relatively rare
!	   IF ( sslo.GE.1 .AND. sshi.LE.line_len ) THEN
!* ***** PRE-SCAN NEIGHBORING POINTS TO FIND QUICK SOLN FOR IRREG AXIS
!	      ibot = isub0 + sslo
!	      itop = isub0 + sshi
!* ... does the last solution still work?
!	      IF ( line_mem(ibot) .LE. wcoord
!     .       .AND. line_mem(itop) .GE. wcoord ) THEN
!	         factor = SNGL(line_mem(itop) - wcoord)
!     .	 		/ SNGL(line_mem(itop) - line_mem(ibot))
!	         RETURN
!* ... is the answer one of the neighboring points?
!	      ELSEIF ( wcoord .LT. line_mem(ibot) ) THEN
!* ... new value is smaller than last time
!	         IF ( wcoord .GT. line_mem(ibot-1) ) THEN
!	            sshi = sslo
!	            sslo = sslo - 1
!	            factor = SNGL(line_mem(ibot) - wcoord)
!     .			   / SNGL(line_mem(ibot) - line_mem(ibot-1))
!	            RETURN
!	         ELSE
!	            itop = ibot
!	            ibot = 1
!	         ENDIF
!	      ELSE
!* ... new value is larger than last time
!	         IF ( wcoord .LT. line_mem(itop+1) ) THEN
!	            sslo = sshi
!	            sshi = sshi + 1
!	            factor = SNGL(line_mem(itop+1) - wcoord)
!     .			   / SNGL(line_mem(itop+1) - line_mem(itop))
!	            RETURN
!	         ELSE
!	            ibot = itop
!	            itop = line_len
!	         ENDIF
!	      ENDIF
!	   ENDIF
* ***** END OF PRE-SCAN NEIGHBORING POINTS

* binary search if axis points are irregularly spaced
* initialize search
	   lo_look  = ibot - 1
	   hi_look  = itop + 1	
	   look     = ( lo_look + hi_look ) / 2

* top of search loop
 100	   wwindex = TM_WORLD(look-isub0, grid, idim, box_middle)
	   IF ( wcoord .GT. wwindex ) THEN

* answer is somewhere above current "look" position
* keep looking at higher values setting this as new low limit for search
	      lo_look = look
	      look = ( look + hi_look + 1 ) / 2

* answer is identically equal to the current "look" position
	   ELSEIF ( wcoord .EQ. wwindex ) THEN
	      point = look - isub0
	      GOTO 1100

	   ELSE
* answer is somewhere below current "look" position
* keep looking at lower values setting this as new high limit for search
	      hi_look = look
	      look = ( look + lo_look ) / 2

	   ENDIF

* have we narrowed it to a 1 point range ?
	   IF ( hi_look - lo_look .LE. 1 ) THEN

* yes - we found it
	      sslo = lo_look - isub0
	      sshi = hi_look - isub0
	      IF ( sslo .EQ. sshi ) THEN
	         factor = 0.0
	      ELSE
	         wwindex = TM_WORLD(hi_look-isub0,grid, idim, box_middle)
	         factor = SNGL(wwindex - wcoord)
     .			/ SNGL(wwindex-
     .			  TM_WORLD(lo_look-isub0,grid, idim, box_middle))
	      ENDIF
	      RETURN
	   ELSE

* no - keep looking
	      GOTO 100
	   ENDIF
	ENDIF

* special conclusions
* ... NO SOLUTION
 1000	sslo = unspecified_int4
	sshi = unspecified_int4
	RETURN

* ... SOLUTION IS SINGLE POINT
 1100	sslo = point
	sshi = point
	factor = 0.0
	RETURN

	END
