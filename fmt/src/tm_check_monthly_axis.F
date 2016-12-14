	SUBROUTINE TM_CHECK_MONTHLY_AXIS (coords, npts, cal_id, t0_string, units, 
     .                                unit_string, start, delta, tunit, true_month)

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
* Check time axis; if it is a monthly axis, with cells the length of the months
* in the given calendar, then we will  define it as a "true_month" calendar.

* coords	coordinate data as read from dataset
* npts		length of coordinate array
* cal_id	calendar ID
* t0_string	time origin
* units		on input, the units of the calendar, 
*                changed on output for true-month axis
* unit_string	on input, the units string for the axis, 
*                changed on output for true-month axis
* start		on input, the first coordinate value, 
*                changed on output for true-month axis
* delta		on input, not set. On output, set for a true-month axis
* tunit		time unit, e.g. 1/12 of seconds in the calendar year
* true_month	flag for true-month calendars
* line_characteristics arrays will also be adjusted

        include 'tmap_errors.parm'
        include 'tmap_dims.parm'
        external xio_data
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
        include 'xunits.cmn_text'
        external xunits_data
        include 'calendar.decl'
        include 'calendar.cmn'

* Argument declarations
	LOGICAL true_month
	INTEGER npts, cal_id, units
	REAL	coords(*), start, delta, tunit
	CHARACTER*(*) t0_string, unit_string

	LOGICAL	TM_DFPEQ_TOL
	INTEGER TM_LENSTR1, STR_SAME, TM_UNIT_ID, TM_UNITS_CAL, 
     .		i, i1, istep, ndx,
     .		slen, year, month, day, zero, status
	REAL	SECS_FROM_BC, TM_SECS_FROM_BC, 
     .		secpday, secs_t0, secs_start, secs_end, secs_box,
     .		secs_box_lo, days_box, days_in_mo, epsilon
	CHARACTER TM_SECS_TO_DATE*20, TM_GET_CALENDAR_NAME*32,
     .		datestring*20, moyr*3
	
  50	FORMAT (7x, I4)

* Initialize
	
	secpday = 24*3600
	epsilon = 1.e-12

	IF (npts .LE. 1) THEN
	   true_month = .FALSE.
	   RETURN
	ENDIF

	true_month = .TRUE.
	
* Seconds from 0000 to t0

	secs_t0 = SECS_FROM_BC( t0_string, cal_id, status )

	DO istep = 1, npts-1

* box low and high, in seconds using box bounds
	   ndx = istep + npts

  	   secs_start = coords(ndx)* un_convert(units)
	   secs_end = coords(ndx+1)* un_convert(units)
	   secs_box = secs_end - secs_start
	   days_box = secs_box/ secpday

* seconds from 0000 to box low 
	   secs_box_lo = secs_t0 + secs_start
	   datestring = TM_SECS_TO_DATE ( secs_box_lo, cal_id )
	
	   moyr = datestring(4:6)
	   month = 0
	   DO i = 1, cals_num_months(cal_id)
	      IF ( STR_SAME(cals_month_names(i, cal_id), moyr) .EQ. 0) THEN
	         month = i
	         EXIT  ! month found, exit from loop
	      ENDIF
	   ENDDO
	   read (datestring, 50) year

	   IF (datestring(1:2) .NE. '01') THEN
	      true_month = .FALSE.
	      GOTO 5000
	   ENDIF

	   day = 1
	   zero = 0
	   
	   secs_start = TM_SECS_FROM_BC ( cal_id,
     .          year, month, day, zero, zero, zero, status)

	   IF (month .LT. 12) THEN
	      month = month + 1
	   ELSE
	      year = year + 1
	      month = 1
	   ENDIF
	   secs_end = TM_SECS_FROM_BC ( cal_id,
     .          year, month, day, zero, zero, zero, status)

           secs_box = secs_end - secs_start
	   days_in_mo = (secs_end - secs_start)/ secpday

	   IF  (.NOT. TM_DFPEQ_TOL ( days_in_mo, days_box, epsilon ) ) THEN
	      true_month = .FALSE.
	      GOTO 5000
	   ENDIF
	ENDDO

* convert first coordinate 
	secs_start = coords(1)* un_convert(units)  ! incoming units
	
* new, monthly units
        units = TM_UNIT_ID ( 'month' )
	units = TM_UNITS_CAL ( units, cal_id, true_month )

	start = (secs_start / un_convert(units) )
	delta = 1.

	datestring = TM_GET_CALENDAR_NAME(cal_id)
	slen = TM_LENSTR1 ( datestring )
	unit_string = 'TRUMONTH ('//datestring(:slen)//')'
	tunit = un_convert( units )

 5000	CONTINUE
	RETURN
	END
