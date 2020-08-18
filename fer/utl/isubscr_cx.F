	INTEGER FUNCTION ISUBSCR_CX ( world_coord, grid, idim, 
     .                                cx ,round_code )

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
* 9/97 *kob*   - NEW ROUTINE
*              - needed in order to implement new method of handling negative
*                time steps.  this routine is nearly identicle to ISUBSCRIPT 
*                except it will use cx_calendar to determine if t dimension
*                is a date or a time step.
* V541: *sh* 2/02 - added support for subspan modulo axes
* V600: *acm*6/06 - fix bug1419. for modulo axes, index at the end of the last 
*                   cell may round up to the next modulo replication 
* V600 *acm* 6/06 - fix declarations, found by solaris compiler
* V612 *acm* 8/08 - Bug 1594. For 64-bit, when translating to within axis 
*                   limits, the comparison IF (wcoord .GT. aend) yielded FALSE 
*                   when in fact the coordinate values should be equal. Use 
*                   TM_FFPEQ to test for equality first.
* V673: *acm* 10/11 Bug 1888. Single-precision calculations in the block starting
*                   * quick sol'n if axis points are regularly spaced
*                   did not give enough accuracy for axes defined in terms of 
*                   seconds.  Convert all this to double-precision calculations.
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. The time 
*                  dimension is now an arg to SECS_TO_TSTEP.
* V683: *acm* 10/12 Fix ticket 1973. If the coordinate value is on the boundary
*                   of the grid cell, precicision could cause the wrong index 
*                   to be returned.
* V683  *acm*10/12 Improve handling of F axes and info about the calendar of the grid.
* V695  *acm* 2/15 cx_calendar stores info about calendar formatting of T and F axes
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* V761  *acm* 7/20 Issue 1981: if coord equals the upper-limit of the uppermost grid 
*                  cell and it's round_down, then this is a modulo op, modulo=-1

        IMPLICIT NONE
	include 'tmap_dims.parm'
#      include "tmap_dset.parm"
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER	grid, idim, round_code, cx
	REAL*8	world_coord

* internal variable declarations:
	LOGICAL	TM_DFPEQ, TM_ITS_SUBSPAN_MODULO, TM_FPEQ_SNGL, 
     .		ITSA_TRUEMONTH_AXIS, is_regular, calendar, true_month
	INTEGER	lo_look, hi_look, axis, ibot, itop, look,
     .		line_len, isubsc, num_mod, round
	REAL*8	SECS_TO_TSTEP, TM_WORLD, TM_WW_AXLEN,
     .		wwindex, wcoord, half_way, start, aend, span, 
     .		first, last, dsubsc, timefac
	REAL*4  delsubsc, half

* initialize
        isubsc = 0
        num_mod = 0

* Distinguish round-down when getting subscripts for a pair lo:hi
* with round-down for a single point

	round = round_code
	IF (round_code .EQ. round_sngl) round = round_dn

* which axis are we on ?
	axis    = grid_line ( idim, grid )
	line_len = line_dim( axis )

* check for peculiar calls
	IF (  axis .EQ. mnormal
     .	 .OR. axis .EQ. munknown ) THEN
	   ISUBSCR_CX = axis
	   RETURN
	ELSEIF ( world_coord .EQ. unspecified_val8 ) THEN
	   ISUBSCR_CX = unspecified_int4
	   RETURN
	ENDIF

* time expressed as seconds since BC ?
	calendar = (idim .EQ. t_dim .OR. idim .EQ. f_dim) .AND. cx_calendar(idim,cx) 
	IF (idim.EQ.f_dim .AND. (line_t0(axis) .EQ. char_init20)) calendar = .FALSE.

	IF ( calendar ) THEN
* no negative needeed on world_coord, because it is not being passed 
* in as negative *kob*  9/97
	   wcoord = SECS_TO_TSTEP( grid, idim, world_coord )
	ELSE
	   wcoord = world_coord
	ENDIF

* special consideration for sub-span modulo axis outside its core range
	IF (TM_ITS_SUBSPAN_MODULO(axis) ) THEN
	   CALL TM_WW_AX_1_N(axis, start, aend)
	   IF ( wcoord.LT.start .OR. wcoord.GT.aend ) THEN
	      line_len = line_len + 1
	      is_regular = .FALSE.
	   ELSE
	      is_regular = line_regular(axis) .AND. .NOT.ITSA_TRUEMONTH_AXIS(axis)

	   ENDIF
	ELSE
	   is_regular = line_regular(axis) .AND. .NOT.ITSA_TRUEMONTH_AXIS(axis)
	ENDIF

* initialize pointers for axes with irregularly spaced points
	IF ( .NOT.is_regular ) THEN
	   ibot  = 1
	   itop  = line_len
	ENDIF

* Modulo axis: translate world coord to within axis range
	IF ( line_modulo( axis ) ) THEN
* ... begin by computing axis limits as outer edges of end boxes
	   span = TM_WW_AXLEN(axis)
	   CALL TM_WW_AXLIMS(axis, start, aend)
	    
	   IF ( ITSA_TRUEMONTH_AXIS(axis) ) THEN 
	      CALL TM_MONTH_TIME_FACTOR (grid, true_month, timefac)
	      start = start * timefac
	      aend = aend * timefac
	      span = span * timefac
	   ENDIF

	   IF (TM_ITS_SUBSPAN_MODULO(axis)) THEN
	      span = line_modulo_len(axis)
	      aend = start + span
	   ENDIF

* ... now translate to within axis limits saving modulo number

           IF (.NOT.TM_DFPEQ(wcoord, aend) .AND. (wcoord .GT. aend)) THEN
	      num_mod = ( wcoord - start ) / span
! 10/92 bug fix: if wcoord is an exact multiple of length (e.g. wccord=24.
!       start=0., span=12.) then modulo remap to top of ax rather than bottom
              IF ( wcoord .EQ. start+num_mod*span )
     .                          num_mod = num_mod - 1
	   ELSEIF ( wcoord .LT. start ) THEN
	      num_mod = ( wcoord - aend  ) / span		! (negative)
	   ELSEIF (TM_DFPEQ(wcoord, start) .AND. round.EQ.round_dn .AND.
     .             round_code.NE.round_sngl .AND.
     .             .NOT.TM_ITS_SUBSPAN_MODULO(axis)) THEN 
	      num_mod = - 1
	   ELSE
	      num_mod = 0
	   ENDIF

	   wcoord = wcoord - num_mod*span
	ENDIF

* get positions of first and last points on axis
        
        IF ( is_regular ) THEN
	   first = line_start( axis )
	   last  = first + line_delta(axis) * (line_len-1) !4/94 was *line_len
        ELSE
           first = TM_WORLD(       1, grid, idim, box_middle)
           last  = TM_WORLD(line_len, grid, idim, box_middle)
	ENDIF

* immediate solutions for given point beyond axis ends
	IF ( wcoord .LE. first .AND. (.NOT.line_modulo(axis)) ) THEN
	   isubsc = 1
	ELSEIF ( wcoord .GE. last .AND. (.NOT.line_modulo(axis)) ) THEN
	   isubsc = line_len

* quick sol'n if axis points are regularly spaced
* Fixing bug 1973 - See if the input was the box_lo or box_hi 
* using a test, is the computed index location halfway between 
* two integers, and if so return the correct index.

	ELSEIF ( is_regular ) THEN
	   dsubsc = (wcoord-first)/line_delta(axis)
	   isubsc = INT( dsubsc )

	   half_way = line_delta(axis)/2.
	   half = 0.5

	   IF ( round .EQ. round_up ) THEN
	      delsubsc = ABS( REAL(dsubsc)-FLOAT(isubsc) )
	      IF ( TM_FPEQ_SNGL(delsubsc, half) )
     .         dsubsc = (wcoord+half_way-first)/line_delta(axis)
	   ENDIF  
	   IF ( round .EQ. round_dn ) THEN
	      delsubsc = ABS( REAL(dsubsc)-FLOAT(isubsc) )
	      IF ( TM_FPEQ_SNGL(delsubsc, half) )
     .         dsubsc = (wcoord-half_way-first)/line_delta(axis)
	   ENDIF  

	   dsubsc = dsubsc + .5  ! to get the right result from INT()
	   isubsc = INT( dsubsc )

	   IF ( TM_DFPEQ(dsubsc,DBLE( isubsc )) ) THEN
* ...... value is on box boundary
	      IF     ( round .EQ. round_up ) THEN
	         isubsc = isubsc + 1
	      ELSEIF ( round .EQ. round_dn ) THEN
	         CONTINUE
	      ELSE
	         STOP 'ISUBSCR_CX round code'
	      ENDIF
	   ELSE
* ...... value is not on box boundary
	      isubsc = isubsc + 1
	   ENDIF

	ELSE
* binary search if axis points are irregularly spaced
* initialize search
	   lo_look  = ibot - 1
	   hi_look  = itop + 1	
	   look     = ( lo_look + hi_look ) / 2

* top of search loop
 100       wwindex = TM_WORLD(look, grid, idim, box_middle)
	   IF ( wcoord .GE. wwindex ) THEN

* answer is somewhere above current "look" position
* keep looking at higher values setting this as new low limit for search
	      lo_look = look
	      look = ( look + hi_look + 1 ) / 2

	   ELSE
* answer is somewhere below current "look" position
* keep looking at lower values setting this as new high limit for search
	      hi_look = look
	      look = ( look + lo_look ) / 2

	   ENDIF

* have we narrowed it to a 1 point range ?
	   IF ( hi_look - lo_look .LE. 1 ) THEN

* yes - we found it - choose box containing it
* ... box boundary between lo_look and hi_look
	      half_way = TM_WORLD(hi_look, grid, idim, box_lo_lim)

* value on box boundary ?
	      IF ( TM_DFPEQ( half_way, wcoord ) ) THEN
	         IF     ( round .EQ. round_up ) THEN
		    isubsc = hi_look
	         ELSEIF ( round .EQ. round_dn ) THEN
		    isubsc = lo_look
	         ELSE
	            STOP 'ISUBSCR_CX round code'
	         ENDIF

* no. value in upper half of box ?
	      ELSEIF ( wcoord .GE. half_way ) THEN
	         isubsc = hi_look

* no. value must be in lower half of box
	      ELSE
	         isubsc = lo_look

	      ENDIF
	   ELSE

* no - keep looking
	      GOTO 100
	   ENDIF
	ENDIF

* add modulo axis correction
	IF ( line_modulo(axis) ) THEN
	   ISUBSCR_CX = isubsc + num_mod*line_len
	ELSE
	   ISUBSCR_CX = isubsc
           IF (isubsc .GT. line_len) ISUBSCR_CX = line_len
	ENDIF
        
* If still not found, apply quick solutions for given point beyond axis ends
        IF (isubsc .EQ. 0) THEN
	   IF ( wcoord .LE. first ) THEN
	      ISUBSCR_CX = 1
	   ELSEIF ( wcoord .GE. last) THEN
	      ISUBSCR_CX = line_len
	   ENDIF
        ENDIF

	RETURN
	END
