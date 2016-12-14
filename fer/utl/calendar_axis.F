	SUBROUTINE CALENDAR_AXIS (idim, grid, cx, status)
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
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*

* Reset time step if axis is a different calendar than Gregorian.
* V683  *acm*10/12 Improve handling of F axes and info about the calendar of the grid.
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	external xunits_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'

* calling argument declarations:
	INTEGER idim, grid, cx, status

* internal variable declarations:
	LOGICAL ITSA_TRUEMONTH_AXIS, true_month
	REAL*8	SECS_FROM_BC, bc_to_t0, ww, new_lo_ww, new_hi_ww, 
     .		new_lo, new_hi, timefac, tcoord
	INTEGER TM_GET_CALENDAR_ID, axis, cal_id, prev_cal_id, istat, ndec
	CHARACTER datestring*20, day_of_mo*2

* axis for this dimension
	axis = grid_line( idim, grid )
* Is it really a calendar axis? Might be an F axis in a grid with a 
* calendar T axis.

	IF (line_t0(axis) .EQ. char_init20) THEN
	   istat = 2
	   GOTO 5000
	ENDIF

	cal_id = TM_GET_CALENDAR_ID ( line_cal_name(axis) )
	bc_to_t0 = SECS_FROM_BC(line_t0(axis), cal_id, istat)
	IF (istat .NE. ferr_ok) GO TO 4000

	prev_cal_id = cx_cal_id(cx)
	IF (prev_cal_id.NE.cal_id .AND. prev_cal_id.NE.unspecified_int4) THEN
	   CALL reset_secs_from_bc (-cx_lo_ww(idim,cx), 
     .		     new_lo_ww, prev_cal_id, cal_id, istat)

	   IF (istat .ne. ferr_ok) GO TO 5000
	   CALL reset_secs_from_bc (-cx_hi_ww(idim,cx), 
     .		     new_hi_ww, prev_cal_id, cal_id, istat)
	   IF (istat.ne. ferr_ok) GO TO 5000
	   cx_cal_id(cx) = cal_id
	ELSE
	   new_lo_ww = -1*cx_lo_ww(idim,cx)
	   new_hi_ww = -1*cx_hi_ww(idim,cx)
	ENDIF

	cx_lo_ww(idim,cx) = (new_lo_ww-bc_to_t0)/line_tunit(axis)
	cx_hi_ww(idim,cx) = (new_hi_ww-bc_to_t0)/line_tunit(axis)

* For a true-month axis, set the coordinate accurately.

	true_month = ITSA_TRUEMONTH_AXIS(axis)

	IF (true_month) THEN
	
	
	   CALL TM_MONTH_TIME_FACTOR (grid, true_month, timefac)

* Hack: Make the request lie inside the cell, if the upper limit is on 
* the boundary. A request for 01-jan-yyy1 to 01-oct-yyy2 should end with
* the grid cell of sep-yyy2, not in the cell that starts at 1-oct-yyy2
* Shift the upper-limit by one second.

           tcoord = (new_lo_ww-bc_to_t0)/line_tunit(axis)
	   CALL TM_TCOORD_TRUMONTH (axis, tcoord, box_middle, new_lo)

* See if the upper limit corresponds to 01-mmm-yyyy 00:00:00
	   ndec = 6
	   tcoord = cx_hi_ww(idim,cx) * timefac
           CALL TRANSLATE_TO_WORLD( tcoord, idim, 
     .                      dgrid_buff, ndec, datestring )
	   day_of_mo = datestring

	   tcoord = (new_hi_ww-bc_to_t0)/line_tunit(axis)
	   IF (day_of_mo .EQ. '01' .AND. 
     .         INDEX(datestring, '00:00:00') .GT. 0) 
     .          tcoord = (new_hi_ww-bc_to_t0-1)/line_tunit(axis)

	   CALL TM_TCOORD_TRUMONTH (axis, tcoord, box_middle, new_hi)

* Convert to days
	   CALL TM_MONTH_TIME_FACTOR (grid, true_month, timefac)

	   cx_lo_ww(idim,cx) = timefac* new_lo
	   cx_hi_ww(idim,cx) = timefac* new_hi
        ENDIF
 

* also fix the world limits for cx_cmnd, if they were originally set 
* using default calendar
* Fix for bug 1365: if the limits for cx_cmnd were not set, do not set them here

	IF ( (cx_lo_ww(idim,cx_cmnd) .NE. unspecified_val8)  .AND.
     .             (cx_cal_id(cx_cmnd) .NE. unspecified_int4)  .AND.
     .             (cal_id .NE. prev_cal_id) ) THEN   
	   cx_lo_ww(idim,cx_cmnd) = -1*new_lo_ww
	   cx_hi_ww(idim,cx_cmnd) = -1*new_hi_ww
	   cx_cal_id(cx_cmnd) = cal_id
	ENDIF

 4000	status = ferr_ok
	RETURN
 5000   status = istat
	RETURN
	END
