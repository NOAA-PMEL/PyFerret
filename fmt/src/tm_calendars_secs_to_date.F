	CHARACTER*(*) FUNCTION TM_CALENDARS_SECS_TO_DATE 
     .      ( calendar_id, num_secs )
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
* Will convert secs since 01-JAN-0000 00:00:00 to a date of the form
* yyyy-mm-dd hh:mm:ss
*
* Written 6/16/87 By Mark Verschell for PMEL/TMAP
*
* revision 0.00 - 06/16/87 - Initial Incarnation
* revision 0.10 - 09/19/88 - changed function to CHAR*(*) from CHAR*20 and now
*			     write to temp string to get around 20 character
*			     minimum length needed for date output
* revision 1.00 - 02/09/89 - Completely rewritten eliminating all if-tests,
*			     loops, and goto's.
* revision 2.00 - 02/20/89 - Rewritten - subtract year 0 - claculate from there
*			     year 0 special case - readded logical leap test
*			     Added some error handling (STOP's)
*
*       J Davison 10.15.99   Copied TM_SECS_TO_DATE() and modified to support
*                            multiple calendars.
* V520 6/00 *sh* - trap attempt to translate a date before 0000


* Argument definition
	REAL*8		num_secs

* Include files
#ifdef unix
	include 'tmap_errors.parm'
        include 'calendar.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
        INCLUDE 'TMAP_PPLV11INC:CALENDAR.CMN'
#endif

*
* Local Definition

	CHARACTER*20	tempstr
	INTEGER		year, month, day, hour,
     .			minute, second, leapadj, cent_cnt, status,
     .			year_cnt, cent4_cnt, year4_cnt, tot_days,
     .			tmp_days
	REAL*8		secs_in_minute, secs_in_hour, secs_in_day,
	1    secs_in_year, secs_in_4years, secs_in_cent,
	2    secs_in_4cents, total_secs, leap_yr_days

	PARAMETER	(secs_in_minute	= 60.0,
	1    secs_in_hour   = secs_in_minute*60.0,
	2    secs_in_day    = secs_in_hour*24.0)

	integer         calendar_id

C       CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

* trap invalid input
	IF ( num_secs .LT. 0 ) THEN
	   TM_CALENDARS_SECS_TO_DATE = '01-JAN-0000:00:00:00'
	   RETURN
	ENDIF

*******************************************************************************
* HANDLE GREGORIAN CALENDAR FIRST  
*******************************************************************************
	if (calendar_id .eq. gregorian) then

*          INITIALIZE
	   leap_yr_days = num_days + 1.0

	   secs_in_year	  = secs_in_day*num_days
	   secs_in_4years = 
	1	secs_in_day*(3*num_days + leap_yr_days)
	   secs_in_cent   = 
	1	secs_in_day*(76*num_days + 24*leap_yr_days)
	   secs_in_4cents = 4*secs_in_cent+secs_in_day
	   total_secs = num_secs

*          SUBTRACT YEAR 0 OFF (IT'S A LEAP YEAR)
	   total_secs = total_secs - secs_in_year - secs_in_day

*          MAKE SURE WE AREN'T WORKING IN YEAR 0
	   IF (total_secs .GE. 0) THEN

*             DECIDE WHICH CENTURY IT IS
	      cent4_cnt = INT(total_secs/secs_in_4cents)
	      total_secs = total_secs - cent4_cnt*secs_in_4cents
	      cent_cnt = INT(total_secs/secs_in_cent)
*             DEC 31 ON LEAP CENTURY TEST (YEARS 400,800,1200...)
	      IF (cent_cnt .EQ. 4 .AND.
	1	   total_secs .GE. cent4_cnt-secs_in_day)   cent_cnt = 3
	      total_secs = total_secs - cent_cnt*secs_in_cent
	      year = 400*cent4_cnt + 100*cent_cnt
	  
*             DECIDE WHAT YEAR IT IS EXACTLY (1 YEAR FOR YEAR 0)
	      year4_cnt = INT(total_secs/secs_in_4years)
	      total_secs = total_secs - year4_cnt*secs_in_4years
	      year_cnt = INT(total_secs/secs_in_year)
*             DEC 31 ON LEAP YEAR TEST (YEARS 0,4,8,12...)
	      IF (year_cnt .EQ. 4 .AND.
	1	   total_secs .GE. secs_in_4years-secs_in_day)   year_cnt = 3
	      total_secs = total_secs - year_cnt*secs_in_year
	      year = year + 4*year4_cnt + year_cnt + 1
	      
	   ELSE
*             Year 0 stuff
	      year = 0
	      total_secs = total_secs + secs_in_year + secs_in_day
	   ENDIF

*          IS THIS A LEAP YEAR? (EVERY 4TH CENTURY IS, OTHERWISE CENTURY ISN'T)
	   leapadj = 0
	   IF (MOD(year,400) .EQ. 0) THEN
	      leapadj = 1
	   ELSE IF (MOD(year,4) .EQ. 0 .AND. MOD(year,100) .NE. 0) THEN
	      leapadj = 1
	   ENDIF

*          HOW  MANY DAYS INTO YEAR IS THIS
	   tot_days = INT(total_secs/secs_in_day)
	   total_secs = total_secs - tot_days*secs_in_day
	   
*          DECIDE WHAT MONTH/DAY IT IS
	   tmp_days = tot_days - leapadj*INT(tot_days/366. + 307./366.)
	   IF (tmp_days .LT. 0) 
	1	STOP 'negative subscript in TM_CALENDARS_SECS_TO_DATE'
	   month = month_by_day(tmp_days+1)
	   leapadj = INT(month/12. +.75)*leapadj
	   day = tot_days - (days_before_month(month)+leapadj) + 1
	   
*          CALCULATE NUMBER OF HOURS
	   hour = INT(total_secs/secs_in_hour)
*          SUBTRACT SECONDS FOR NUMBER OF HOURS
	   total_secs = total_secs - secs_in_hour*(hour)
*          CALCULATE NUMBER OF MINUTES
	   minute = INT(total_secs/secs_in_minute)
*          SUBTRACT SECONDS FOR NUMBER OF MINUTES
	   total_secs = total_secs - secs_in_minute*(minute)
*          CALCULATE NUMBER OF SECONDS
	   second = total_secs

******************************************************************************
* ALL OTHER SUPPORTED AND REGULAR CALENDARS
******************************************************************************
	else if (calendar_id .eq. common   .or.
     .           calendar_id .eq. leap_yr  .or. 
     .           calendar_id .eq. days_360 .or. 
     .           calendar_id .eq. tmap_360 .or. 
     .           calendar_id .eq. peculiar) then

*          INITIALIZE
	   secs_in_year	= secs_in_day*num_days
	   secs_in_cent = 100*secs_in_year
	   total_secs = num_secs

*          SUBTRACT YEAR 0 OFF
	   total_secs = total_secs - secs_in_year

*          MAKE SURE WE AREN'T WORKING IN YEAR 0
	   IF (total_secs .GE. 0) THEN

*             DECIDE WHICH CENTURY IT IS
	      cent_cnt = INT(total_secs/secs_in_cent)
	      total_secs = total_secs - cent_cnt*secs_in_cent
	      year = 100*cent_cnt
	  
*             DECIDE WHAT YEAR IT IS EXACTLY (1 YEAR FOR YEAR 0)
	      year_cnt = INT(total_secs/secs_in_year)
	      total_secs = total_secs - year_cnt*secs_in_year
	      year = year + year_cnt + 1
	      
	   ELSE
*             Year 0 stuff
	      year = 0
	      total_secs = total_secs + secs_in_year
	   ENDIF

*          HOW  MANY DAYS INTO YEAR IS THIS
	   tot_days = INT(total_secs/secs_in_day)
	   total_secs = total_secs - tot_days*secs_in_day
	   
*          DECIDE WHAT MONTH/DAY IT IS
	   month = month_by_day(tot_days+1)
	   day = tot_days - days_before_month(month) + 1
	   
*          CALCULATE NUMBER OF HOURS
	   hour = INT(total_secs/secs_in_hour)
*          SUBTRACT SECONDS FOR NUMBER OF HOURS
	   total_secs = total_secs - secs_in_hour*(hour)
*          CALCULATE NUMBER OF MINUTES
	   minute = INT(total_secs/secs_in_minute)
*          SUBTRACT SECONDS FOR NUMBER OF MINUTES
	   total_secs = total_secs - secs_in_minute*(minute)
*          CALCULATE NUMBER OF SECONDS
	   second = total_secs
	else
	   stop 'Illegal calendar in tm_calendars_secs_to_date()'
	end if

* Put it in a string
	WRITE (tempstr,1000,ERR=5000) day, month_names(month), 
     .      year, hour, minute, second

1000	FORMAT(I2.2,'-',A3,'-',I4.4,' ',2(I2.2,':'),I2.2)
	TM_CALENDARS_SECS_TO_DATE = tempstr
	GOTO 9990

* Date syntax error
 5000	CALL TM_ERRMSG (merr_erreq, status, 'TM_CALENDARS_SECS_TO_DATE',
     .			no_descfile, no_stepfile,
     .			no_errstring, no_errstring, *9000)
 9000	STOP

* finished
9990	RETURN
	END
