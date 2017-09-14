	RECURSIVE FUNCTION TM_WORLD_AX( isubscript, iaxis,
     .					where_in_box ) RESULT(world_ax_result)

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
* convert a subscript value on an axis into a "world" coordinate
* position within the grid box corresponding to the given subscript is
* determined by the value where_in_box

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V530 10/00 *sh* - extracted from tm_world so that it could be called from
*		    routines that know the axis but not a container grid
* V533 6/01 *sh* - using TM_AXIS_RECURSIVE
*      10/01 *kob* - change TM_AXIS_RECURSIVE to proper logical type
* V541 2/02 *sh* - added support for subspan modulo axes
*		 - slight change to calculation for all modulo axes
* V542 10/02 *sh* - bug fix for irregular, subspan modulo
* V542 10/02 *sh* - bug fix for irregular, subspan modulo
* V71  12/16 *acm* Ticket 2158: working with true monthly time axes.
*                  call tm_tcoord_trumonth to return the correct
*                  coordinate value for irregular months corresponding 
*                  to the calendar

* Include files
#include "gt_lib.parm"
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	external xunits_data
	include 'implicit.parm'

* calling argument declarations:
	REAL*8  world_ax_result
	INTEGER	isubscript, iaxis, where_in_box

* internal variable declarations:
	LOGICAL	TM_AXIS_RECURSIVE, TM_ITS_SUBSPAN_MODULO,
     .		ITSA_TRUEMONTH_AXIS, recursive, true_month
	INTEGER	TM_MODULO_LINE_DIM, isub, line_len, rmod, new_ss, 
     .		lo_ss, hi_ss, parent_len, new_where
	REAL*8	TM_WORLD_RECUR, TM_MODULO_AXLEN, GET_LINE_COORD,
     .		midpoint, box_size, tempwld, lo, hi

* initialize
	line_len = TM_MODULO_LINE_DIM( iaxis )
	true_month = ITSA_TRUEMONTH_AXIS(iaxis)

* if this is a child (e.g. every Nth point) of an irregularly-spaced
* axis then use a recursive C routine to find the answer
	recursive = iaxis .GT. max_lines
	IF ( recursive ) 
     .    recursive = TM_AXIS_RECURSIVE(iaxis, true_month)

	IF (recursive) THEN 
	world_ax_result = unspecified_val8  ! default

	IF (line_class(iaxis) .EQ. PLINE_CLASS_STRIDE) THEN 
* in irreg axis striding the box edges cannot simply be read from
* the box edge array (think about it) So we have xtra logic here.

	new_ss = INT(line_start(iaxis)) + (isubscript-1)* INT(line_delta(iaxis))
	IF ( line_regular(iaxis) .OR. where_in_box.EQ.BOX_MIDDLE ) THEN 

	  world_ax_result = TM_WORLD_AX (new_ss, line_parent(iaxis),
     .					where_in_box)
	    
* Extra logic for irregular axis strides
	ELSE  
* cases to consider: interpolate to neighbor above or below
* use lower or upper limit of entire axis

	  IF (where_in_box .EQ.  BOX_LO_LIM) THEN
	    lo_ss = INT(line_start(iaxis))
     .		    + (isubscript-2)* INT(line_delta(iaxis))
	    hi_ss = new_ss
	  ELSE
	    lo_ss = new_ss
	    hi_ss = INT(line_start(iaxis))
     .		     + (isubscript-0)* INT(line_delta(iaxis))
	  ENDIF

	  parent_len = line_dim( line_parent(iaxis))
	  IF ( line_modulo( line_parent(iaxis) ) 
     .	       .OR. (lo_ss.GE.1 .AND. hi_ss.LE.parent_len) ) THEN ! interpolate 
	    new_where = BOX_MIDDLE
	    world_ax_result = 0.5 * 
     .		   ( TM_WORLD_AX (lo_ss, line_parent(iaxis),new_where) + 
     .		     TM_WORLD_AX (hi_ss, line_parent(iaxis),new_where) )

	  ELSE IF (where_in_box .EQ.  BOX_LO_LIM) THEN  ! lower axis edge  
	    new_ss = 1
	    world_ax_result = TM_WORLD_AX (new_ss, line_parent(iaxis), where_in_box)
	  ELSE ! upper axis edge 
	    new_ss = parent_len
	    world_ax_result = TM_WORLD_AX (new_ss, line_parent(iaxis), where_in_box)
	  ENDIF

* Fix for getting data at ends when subspan modulo strides, irregular axis.
	  
	  IF ( TM_ITS_SUBSPAN_MODULO(line_parent(iaxis)) .AND. 
     .	       line_modulo(line_parent(iaxis)) .AND. 
     .	       parent_len.GT.0 ) THEN
            IF (lo_ss .EQ. 0 .OR. lo_ss .EQ. -1) THEN
	      new_where = BOX_LO_LIM
	      lo_ss = 1
	      world_ax_result = 
     .		   TM_WORLD_AX (lo_ss, line_parent(iaxis),new_where)

            ELSE IF (hi_ss .EQ. parent_len+1) THEN
	      new_where = BOX_HI_LIM
	      hi_ss = parent_len
	      world_ax_result = 
     .		   TM_WORLD_AX (hi_ss, line_parent(iaxis),new_where)
            ENDIF

	  ENDIF

	ENDIF 

! end of recursive for strides case
	ELSE IF (line_class(iaxis) .EQ. PLINE_CLASS_MIDPT) THEN
	   world_ax_result = TM_WORLD_AX(isubscript, iaxis, where_in_box)

	ENDIF ! line_class: stride, or midpoint

	RETURN
	ENDIF  ! end of recursive  recursive 

* force given subsc to data range as appropriate for modulo or non-modulo axes
	IF ( line_modulo( iaxis ) ) THEN
	  isub = MOD(isubscript-1,line_len) + 1
	  IF (isub .LE. 0) isub = isub + line_len
	ELSE
	  isub = MIN( line_len, MAX( 1, isubscript ) )
	ENDIF

	IF ( TM_ITS_SUBSPAN_MODULO(iaxis) .AND. isub.EQ.line_len ) THEN
* given index falls in the "void" region of a subspan modulo axis
* ... get the box_hi_lim of the Nth point in the core region
	  CALL TM_WW_AXLIMS(iaxis, lo, hi)
* ... now where within the grid box ?
	  IF ( where_in_box .EQ. box_lo_lim ) THEN
	    tempwld = hi
	  ELSEIF ( where_in_box .EQ. box_middle ) THEN
	    tempwld = ((lo+line_modulo_len(iaxis)) + hi) / 2.
	  ELSE
	    tempwld = lo + line_modulo_len(iaxis)
	  ENDIF
	  IF (isubscript .LE. 0) THEN
	    rmod = isubscript/line_len - 1
	  ELSE
	    rmod = (isubscript-1)/line_len
	  ENDIF
	  world_ax_result = tempwld + rmod* line_modulo_len(iaxis)

	ELSEIF ( line_regular( iaxis ) ) THEN
* regularly spaced points
* ... calculate midpoint and box_size values
	  midpoint = line_start(iaxis) + (isub-1)* line_delta(iaxis)
	  box_size = line_delta(iaxis)
* ... now where within the grid box ?
	  IF ( where_in_box .EQ. box_lo_lim ) THEN
	    tempwld = midpoint - ( box_size / 2. )
	  ELSEIF ( where_in_box .EQ. box_middle ) THEN
	    tempwld = midpoint
	  ELSE
	    tempwld = midpoint + ( box_size / 2. )
	  ENDIF

* Was it at true-month axis?  If so, adjust. 

	  IF (true_month) CALL TM_TCOORD_TRUMONTH (iaxis, 
     .			  midpoint, where_in_box, tempwld)

	  IF ( line_modulo( iaxis ) ) THEN
	    IF (isubscript .LE. 0) THEN
	      rmod = isubscript/line_len - 1
	    ELSE
	      rmod = (isubscript-1)/line_len
	    ENDIF
	    world_ax_result = tempwld + rmod*TM_MODULO_AXLEN(iaxis)
	  ELSE
	    world_ax_result = tempwld
	  ENDIF

	ELSE
* irregularly spaced points

* ... now where within the grid box ?
	  IF ( where_in_box .EQ. box_lo_lim ) THEN
	    tempwld = GET_LINE_COORD (lineedg(iaxis)%ptr, isub)
	  ELSEIF ( where_in_box .EQ. box_middle ) THEN
	    tempwld =  GET_LINE_COORD (linemem(iaxis)%ptr, isub)
	  ELSE
	    tempwld = GET_LINE_COORD (lineedg(iaxis)%ptr, isub+1)
	  ENDIF

	  IF ( line_modulo( iaxis ) ) THEN
	    IF (isubscript .LE. 0) THEN
	      rmod = isubscript/line_len - 1
	    ELSE
	      rmod = (isubscript-1)/line_len
	    ENDIF
	    world_ax_result = tempwld + rmod*TM_MODULO_AXLEN(iaxis)
	  ELSE
	    world_ax_result = tempwld
	  ENDIF

	ENDIF

	RETURN
	END
