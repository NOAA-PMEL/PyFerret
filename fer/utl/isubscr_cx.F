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

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	grid, idim, round_code, cx
	REAL*8	world_coord

* internal variable declarations:
	LOGICAL	TM_FPEQ
	INTEGER	lo_look, hi_look, iaxis, ibot, itop, look,
     .		line_len, isubsc, num_mod
	REAL 	subsc
	REAL*8	SECS_TO_TSTEP, TM_WORLD, 
     .		wwindex, wcoord, half_way, start, aend, span, first, last

* which axis are we on ?
	iaxis    = grid_line ( idim, grid )
	line_len = line_dim( iaxis )

* check for peculiar calls
	IF (  iaxis .EQ. mnormal
     .	 .OR. iaxis .EQ. munknown ) THEN
	   ISUBSCR_CX = iaxis
	   RETURN
	ELSEIF ( world_coord .EQ. unspecified_val8 ) THEN
	   ISUBSCR_CX = unspecified_int4
	   RETURN
	ENDIF

* time expressed as seconds since BC ?
	IF ( idim .EQ. t_dim .AND. cx_calendar(cx) ) THEN
* no negative needeed on world_coord, because it is not being passed 
* in as negative *kob*  9/97
	   wcoord = SECS_TO_TSTEP( grid, world_coord )
	ELSE
	   wcoord = world_coord
	ENDIF

* initialize pointers for axes with irregularly spaced points
	IF ( .NOT.line_regular( iaxis ) ) THEN
	   ibot  = 1
	   itop  = line_len
	ENDIF

* Modulo axis: translate world coord to within axis range
	IF ( line_modulo( iaxis ) ) THEN
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

* get positions of first and last points on axis
	IF ( line_regular( iaxis ) ) THEN
	   first = line_start( iaxis )
	   last  = first + line_delta(iaxis) * (line_len-1) !4/94 was *line_len
        ELSE
           first = TM_WORLD(       1, grid, idim, box_middle)
           last  = TM_WORLD(line_len, grid, idim, box_middle)
	ENDIF

* immediate solutions for given point beyond axis ends
	IF ( wcoord .LE. first ) THEN
	   isubsc = 1
	ELSEIF ( wcoord .GE. last) THEN
	   isubsc = line_len

* quick sol'n if axis points are regularly spaced
	ELSEIF ( line_regular( iaxis ) ) THEN
	   subsc = SNGL( (wcoord-first)/line_delta(iaxis) ) + .5
	   isubsc = INT( subsc )
	   IF ( subsc .EQ. FLOAT( isubsc ) ) THEN
* ...... value is on box boundary
	      IF     ( round_code .EQ. round_up ) THEN
	         isubsc = isubsc + 1
	      ELSEIF ( round_code .EQ. round_dn ) THEN
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
	      IF ( TM_FPEQ( SNGL(half_way), SNGL(wcoord) ) ) THEN
	         IF     ( round_code .EQ. round_up ) THEN
		    isubsc = hi_look
	         ELSEIF ( round_code .EQ. round_dn ) THEN
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
	IF ( line_modulo(iaxis) ) THEN
	   ISUBSCR_CX = isubsc + num_mod*line_len
	ELSE
	   ISUBSCR_CX = isubsc
	ENDIF

	RETURN
	END
