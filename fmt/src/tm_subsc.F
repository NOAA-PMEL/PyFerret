	INTEGER FUNCTION TM_SUBSC ( world_coord, grid, idim, clue )

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
* convert a "world" coordinate into the subscript value on an axis by finding
* the grid box that contains the world coordinate.
* A clue may be provided to speed processing of irregular axes

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  12/9/90 - based on FERRET's TM_SUBSC version 2.00
* NOTE: as of 9/90 regular axes have NOT been tested
* 1/95 *sh* inserted support for modulo axes from FERRET's ISUBSCRIPT
* 10/95 - use TM_WORLD instead of direct line_mem to support recursive cases
*  1/96 - many references to isub0 remained despite 10/95 changes
* *kob* 10/96   - Linux port.  Linux didn't like tabs between "include"
*                 and what was being included.  removed them
* V542 2/02 *sh* - added support for modulo length > axis length
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files
* V71  3/16 *acm* Ticket 2158: working with true monthly time axes.
* V71  1/17 *acm* Ticket 2496: make the result index consistent with the
*                 result form isubscr_cx when incoming coordinate value
*                 is ON a cell boundary

* calling arguments:
* world_coord	- coordinate value to be translated to subscript
* grid		- underlying grid number
* idim		- underlying axis number (1-nferdims)
* clue		- performance assist to code - check this as possible result

	include 'tmap_dims.parm'
#include "gt_lib.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data


* calling argument declarations:
	INTEGER	grid, idim, clue
	REAL*8	world_coord

* internal variable declarations:
	LOGICAL	TM_ITS_SUBSPAN_MODULO, ITSA_TRUEMONTH_AXIS, TM_DFPEQ, regular
	INTEGER	lo_look, hi_look, ibot, itop, look, axis,
     .		line_len, isubsc
	INTEGER	num_mod
	REAL 	subsc
	REAL*8	TM_WORLD, TM_WW_AXLEN,
     .		half_way, start, aend, span, first, last, wcoord, wwindex

* we need to be able to modify the search value for modulo axes
	wcoord = world_coord

* which axis are we on ?
	axis    = grid_line ( idim, grid )
	line_len = line_dim( axis )

* special consideration for sub-span modulo axis outside its core range
	IF (TM_ITS_SUBSPAN_MODULO(axis) ) THEN
	   CALL TM_WW_AX_1_N(axis, start, aend)
	   IF ( wcoord.LT.start .OR. wcoord.GT.aend ) THEN
	      line_len = line_len + 1
	      regular = .FALSE.
	   ELSE
	      regular = line_regular(axis) .AND. .NOT.ITSA_TRUEMONTH_AXIS(axis)
	   ENDIF
	ELSE
	   regular = line_regular(axis) .AND. .NOT.ITSA_TRUEMONTH_AXIS(axis)
	ENDIF

* initialize pointers for axes with irregularly spaced points
	IF ( .NOT.regular ) THEN
	   IF ( clue.GE.1 .AND. clue.LE.line_len ) THEN
	      wwindex = TM_WORLD(clue, grid, idim, box_middle)
* ... check clue - (clue accepted only for exact equality of coordinates)
	      IF ( wcoord .EQ. wwindex ) THEN
	         TM_SUBSC = clue
	         RETURN
	      ENDIF
	   ENDIF
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

* ... now translate wcoord to within axis limits saving modulo number
           IF     ( wcoord .GT. aend  ) THEN
              num_mod = ( wcoord - start ) / span
! 10/92 bug fix: if wcoord is an exact multiple of length (e.g. wccord=24.
!       start=0., span=12.) then modulo remap to top of ax rather than bottom
              IF ( wcoord .EQ. start+num_mod*span )
     .                          num_mod = num_mod - 1
           ELSEIF ( wcoord .LT. start ) THEN
              num_mod = ( wcoord - aend  ) / span               ! (negative)
           ELSE
              num_mod = 0
           ENDIF
           wcoord = wcoord - num_mod*span
        ENDIF

* get positions of first and last points on axis
	IF ( regular ) THEN
	   first = line_start( axis )
	   last  = first + line_delta( axis ) * line_len
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
	   subsc  = ( (wcoord-first)/line_delta(axis) ) + .5
	   isubsc = INT( subsc ) + 1		! round up if ambiguous

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
* Change the logic if the point is on the boundary, returning the lower index.
* This is consistent with what isubscr_cx does.

* ... box boundary between lo_look and hi_look
	      half_way = TM_WORLD(hi_look, grid, idim, box_lo_lim)

* value on box boundary ?
	      IF ( TM_DFPEQ( half_way, wcoord ) ) THEN
	         isubsc = lo_look
	      ELSE

* no. value in upper half of box ?
	         IF ( wcoord .GT. half_way ) THEN
	            isubsc = hi_look

* no. value must be in lower half of box (or on the box boundary)
	         ELSE
	            isubsc = lo_look

	         ENDIF
	      ENDIF
	   ELSE

* no - keep looking
	      GOTO 100
	   ENDIF
	ENDIF

* insert modulo axis correction
        IF ( line_modulo(axis) ) THEN
           TM_SUBSC = isubsc + num_mod*line_len
        ELSE
           TM_SUBSC = isubsc
        ENDIF

	RETURN
	END
