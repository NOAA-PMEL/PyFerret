	INTEGER FUNCTION ISUBSCRIPT ( world_coord, grid, idim, round_code )

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
* convert a "world" coordinate into the subscript value on an axis of the grid
* box containing this value.  When the coordinate lies directly on the boundary
* use round_code to resolve the ambiguity.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/16/86
* revision 1.0 - 3/10/86 - grid lines stored in COMMON/~VARIABLES/
* revision 1.2 - 3/23/87 - added new TMAP library: time axes, file var. names
* revision 1.3 - 2/18/88 - incorporated "regular" axes; trapped "unspecified"
*			    world coordinate input
* revision 1.4 - 7/28/88 - interpret negative times as seconds since 1-JAN-0000
* revision 1.5 -12/14/88 - incorporate new IO library with grid box lower bounds
*			   instead of box sizes
* revision 2.0 - 2/3/89  - incorporated modulo axes and reorganized
* Unix/RISC port - 1/91 - cant use "/star" in comment lines on Unix
*                  7/91 - eliminated unneeded common *variables and errmsg.parm
*                 10/92 - bug fix for modulo axis point that lies exactly on a
*                         multiple of the axis length (search "10/92")
* V301: 2/7/94 - bug in calculation of "last"for regular axes.  This bug never
*		 showed up in program behavior - possibly because
*		 GRID_WORLD_EXTREMES is always called first (?)
* V420: 10/95 - for axes which are children of others avoid direct use of
*		line_mem -- instead use TM_WORLD
*	 1/96 - bug fix: many references to sub0 remained despite 10/95 chgs
* 9/97 *kob* - w/ negative time axis capablities, this routine will not have
*	      to worry about date strings...
* V541: *sh* 2/02 - added support for subspan modulo axes

	include 'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'

* calling argument declarations:
	INTEGER	grid, idim, round_code
	REAL*8	world_coord

* internal variable declarations:
	LOGICAL	TM_FPEQ, TM_ITS_SUBSPAN_MODULO, regular
	INTEGER	lo_look, hi_look, axis, ibot, itop, look,
     .		line_len, isubsc, num_mod
	REAL 	subsc
	REAL*8	TM_WORLD, TM_WW_AXLEN, 
     .		wwindex, wcoord, half_way, start, aend, span, first, last

* which axis are we on ?
	axis    = grid_line ( idim, grid )
	line_len = line_dim( axis )

* check for peculiar calls
	IF (  axis .EQ. mnormal
     .	 .OR. axis .EQ. munknown ) THEN
	   ISUBSCRIPT = axis
	   RETURN
	ELSEIF ( world_coord .EQ. unspecified_val8 ) THEN
	   ISUBSCRIPT = unspecified_int4
	   RETURN
	ENDIF

	   wcoord = world_coord

* special consideration for sub-span modulo axis outside its core range
	IF (TM_ITS_SUBSPAN_MODULO(axis) ) THEN
	   CALL TM_WW_AX_1_N(axis, start, aend)
	   IF ( wcoord.LT.start .OR. wcoord.GT.aend ) THEN
	      line_len = line_len + 1
	      regular = .FALSE.
	   ELSE
	      regular = line_regular( axis )
	   ENDIF
	ELSE
	   regular = line_regular( axis )
	ENDIF

* initialize pointers for axes with irregularly spaced points
	IF ( .NOT.regular ) THEN
	   ibot  = 1
	   itop  = line_len
	ENDIF

* Modulo axis: translate world coord to within axis range
	IF ( line_modulo( axis ) ) THEN
* ... begin by computing axis limits as outer edges of end boxes
	   span = TM_WW_AXLEN(axis)
	   CALL TM_WW_AXLIMS(axis, start, aend)
	   IF (TM_ITS_SUBSPAN_MODULO(axis)) THEN
	      span = line_modulo_len(axis)
	      aend = start + span
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
	IF ( regular ) THEN
	   first = line_start( axis )
	   last  = first + line_delta(axis) * (line_len-1) !4/94 was *line_len
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
	ELSEIF ( regular ) THEN
	   subsc = SNGL( (wcoord-first)/line_delta(axis) ) + .5
	   isubsc = INT( subsc )
	   IF ( subsc .EQ. FLOAT( isubsc ) ) THEN
* ...... value is on box boundary
	      IF     ( round_code .EQ. round_up ) THEN
	         isubsc = isubsc + 1
	      ELSEIF ( round_code .EQ. round_dn ) THEN
	         CONTINUE
	      ELSE
	         STOP 'ISUBSCRIPT round code'
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
	            STOP 'ISUBSCRIPT round code'
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
	   ISUBSCRIPT = isubsc + num_mod*line_len
	ELSE
	   ISUBSCRIPT = isubsc
	ENDIF

	RETURN
	END
