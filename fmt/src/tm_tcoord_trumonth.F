	SUBROUTINE TM_TCOORD_TRUMONTH (iaxis, monthval, where_in_box, 
     .		                       tempwld)
 
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
* For a true-month axis, return the exact coordinate value for irregular months 
* corresponding to the calendar, so that the grid cells lie exactly one month apart.
* Input and output are in units of months.
*
* V71 12/16 *acm* Ticket 2158: working with true monthly time axes.

* Arguments
*  iaxis	axis number
*  monthval	coordinate value (on regularly spaced axis)
*  where_in_box	flag for location to return
*  tempwld	On output, return correct coordinate value on true 
*               monthly axis for lo, mid, or hi box location.

* Include files
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
	include 'xunits.cmn_text'
	external xunits_data
        include 'calendar.decl'
        include 'calendar.cmn'
 
* calling argument declarations:
	INTEGER iaxis, where_in_box
	REAL	monthval, tempwld

* internal variable declarations:
	LOGICAL	TM_DIGIT_TEST
	INTEGER TM_GET_CALENDAR_ID, STR_SAME, TM_LENSTR1, TM_UNIT_ID,
     .		i, i1, i2, idim, cal_id, ndec, status, 
     .		yrt0, mot0, dat0, ht0, mit0, set0,
     .		yr1, mo1, da1, hr1, mi1, se1, 
     .		yr2, mo2, da2, hr2, mi2, se2
	REAL	TM_SECS_FROM_BC, SECS_FROM_BC, secpday, edge_lo, edge_hi,   
     .		secs0,secs1, secs2, days_t0, days_yrs, days_this_year, 
     .		days_all_to_lo, days_cell, tcoord,
     .		num_secs, start_secs, offset_secs

	!print *, 'TM_TCOORD_TRUMONTH faster version'

* Initialize.

	edge_lo = 0.
	edge_hi = 0.
	secs0 = 0.
	secs1 = 0.
	secs2 = 0.
	days_t0 = 0.
	days_all_to_lo = 0.
	days_cell = 0.

	IF (line_t0(iaxis) .EQ. char_init) GOTO 5000  ! unformatted dataset

        idim = t_dim
	secpday = 24*3600

	cal_id = TM_GET_CALENDAR_ID ( line_cal_name(iaxis) )
        CALL TM_BREAK_DATE (line_t0(iaxis), cal_id, yrt0, mot0, dat0,
     .                      ht0, mit0, set0, status)

* dgrid_buff is a fake grid to use, defined in tmap_dims.parm

        grid_line(idim, dgrid_buff) = iaxis

* first coordinate value on the axis start with days, get date

	tcoord = line_start(iaxis) * line_tunit(iaxis)/ un_convert(pun_day)

* Find the number of seconds since start of data set for this time step

	start_secs = SECS_FROM_BC( line_t0(iaxis), cal_id, status )
        offset_secs = tcoord* un_convert(pun_day) 
	num_secs = start_secs + offset_secs

	CALL TM_SECS_TO_YMDHMS(num_secs, cal_id, yr1, mo1, da1,
     .                      hr1, mi1, se1, status)

!        print *, 'line start info yr1 etc', yr1, mo1, da1,
!    .                      hr1, mi1, se1

* Monthval is in months. Translate to days and get date string

	tcoord = monthval * line_tunit(iaxis)/ un_convert(pun_day)

* First find the offset from 01-JAN-0000 00:00:00 from the descriptor file
	start_secs = TM_SECS_FROM_BC (cal_id, yrt0, mot0, dat0,
     .                      ht0, mit0, set0, status)

* Find the number of seconds since start of data set for this time step
        offset_secs = tcoord* un_convert(pun_day) 
	num_secs = start_secs + offset_secs

	CALL TM_SECS_TO_YMDHMS(num_secs, cal_id, yr2, mo2, da2,
     .                      hr2, mi2, se2, status)
!	print *, 'coord info yr2 etc', yr2, mo2, da2,
!     .                      hr2, mi2, se2

* Time corresponding to the lower edge of cell, year-month with 
* day 1 and zero for hr,min,sec

	da2 = 1
	hr2 = 0
	edge_lo = TM_SECS_FROM_BC (cal_id, yr2, mo2, da2, hr2, hr2, hr2, status)
        !print *, 'edge_lo', edge_lo

* Correct days from T0 to end-of-year in which T0 lies

* seconds since 1-jan-0000 to t0

	days_t0 = 0
	IF (yr2 .GT. yrt0) THEN
           secs1 = start_secs

* seconds since 1-jan-0000 to end of year of t0
           secs2 = TM_SECS_FROM_BC (cal_id, yrt0+1, 1, 1, 0, 0, 0, status)

* ... convert to days
	   days_t0 = (secs2-secs1)/ secpday
	   !print *, 'secs1, secs2, days_t0', secs1, secs2, days_t0
	ENDIF

* ... days in full years between the end of t0 year and start of coord year

	days_yrs = 0
	secs1 = 0
        IF (yr2 .GT. yrt0+1) THEN

	   secs0 = TM_SECS_FROM_BC (cal_id, yrt0+1, 1, 1, 0, 0, 0, status)

	   secs1 = TM_SECS_FROM_BC (cal_id, yr2, 1, 1, 0, 0, 0, status)
	
* ... convert to days
	   days_yrs = (secs1 - secs0)/ secpday
	   !print *, 'secs1, secs0, days_yrs', secs1, secs0, days_yrs

	ENDIF

* ... days from start of coordinate year to lower limit of box

	IF (secs1 .EQ. 0) secs1 = TM_SECS_FROM_BC (cal_id, yr2, 1, 1, 0, 0, 0, status)

	days_this_year = (edge_lo-secs1)/ secpday

	days_all_to_lo = days_t0 + days_yrs + days_this_year
!	print *, 'days_this_year, days_all_to_lo', 
!     .    days_this_year, days_all_to_lo

        IF (where_in_box .EQ. box_lo_lim) THEN  ! we are done

           tempwld = days_all_to_lo

        ELSE

* Get the correct time coordinate at the upper limit of the box.

           IF (mo2 .LT. 12) THEN
	      edge_hi = TM_SECS_FROM_BC (cal_id, yr2, mo2+1, 1, 0, 0, 0, status)
	   ELSE
	      edge_hi = TM_SECS_FROM_BC (cal_id, yr2+1, 1, 1, 0, 0, 0, status)
	   ENDIF
cc	   edge_hi = edge_hi - 1  ! the top of the box, not the bottom of the next one.

	   days_cell = (edge_hi - edge_lo)/ secpday

* Put the result halfway between the corrected high and low
           IF (where_in_box .EQ. box_middle) THEN

	      tempwld = (days_all_to_lo + days_cell/2.)

* Or return the high coordinate
	   ELSE IF (where_in_box .EQ. box_hi_lim) THEN

	      tempwld = (days_all_to_lo + days_cell)

	   ENDIF

	ENDIF

* We have the coordinate in days, return it in months.

        secs0 = un_convert( line_unit_code(iaxis) ) 
!	print *, 'tempwld = tempwld* secpday/ secs0',
!     .    tempwld, secpday, secs0, tempwld* secpday/ secs0
        tempwld = tempwld* secpday/ secs0

5000    RETURN
	END
