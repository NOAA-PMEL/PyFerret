	REAL*8 FUNCTION TM_CALENDARS_SECS_FROM_BC ( calendar_id,
     .    year, month, day, hour, minute, second)
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
* Will convert a date of yyyy-mm-dd hh:mm:ss to seconds since
* 0000-01-01 00:00:00
*
* Written 6/16/87 By Mark Verschell for PMEL/TMAP
*
* revision 0.00 - 06/16/87 - Initial Incarnation
*
*       J Davison 10.20.99   Copied TM_SECS_FROM_BC() and modified to support
*                            multiple calendars.

* Include files
#ifdef unix
        include 'calendar.cmn'
#else
        INCLUDE 'TMAP_PPLV11INC:CALENDAR.CMN'
#endif

* Argument definition

	INTEGER         calendar_id
	INTEGER		year, month, day, hour, minute, second
*
* Local Definition
	REAL*8		secs_in_minute, secs_in_hour, secs_in_day,
	1    secs_in_year, secs_in_century, total_secs,
	2    leap_yr_days

	PARAMETER	(secs_in_minute	 = 60.0,
	1    secs_in_hour	 = secs_in_minute*60.0,
	2    secs_in_day	 = secs_in_hour*24.0)


*****************************************************************************
*****************************************************************************


*****************************************************************************
* HANDLE GREGORIAN CALENDAR FIRST
*****************************************************************************
	if (calendar_id .eq. gregorian) then

*	   INITIALIZE
	   leap_yr_days = num_days + 1.0

	   secs_in_year	 = secs_in_day*num_days
	   secs_in_century = 
	1	secs_in_day*(76*num_days + 24*leap_yr_days)

*          ADD LOTS OF SECONDS FOR EACH CENTURY SINCE 0 DAY
	   total_secs = secs_in_century * INT(year/100)

*          ADD A DAY FOR EVERY 400 YEARS (LEAP YEAR AT CENTENNIAL)
	   total_secs = total_secs + secs_in_day*INT(year/400.+0.9975)

*          ADD A YEAR FOR EACH YEAR SINCE TURN OF CENTURY
	   total_secs = total_secs + secs_in_year*MOD(year,100)

*          ADD A DAY FOR EACH LEAP YEAR SINCE CENTENNIAL(EXCEPT FOR CENTENNIAL)
	   total_secs = 
	1	total_secs + secs_in_day*INT((MOD(year,100)-1)/4)

*          ADD SECONDS FOR NUMBER OF MONTHS
	   total_secs = 
	1	total_secs + days_before_month(month)*secs_in_day

*          ADD 1 DAY WORTH OF SECONDS IF THIS IS LEAP YEAR AND PAST FEBRUARY
	   if (month .GT. 2) then
	      if (MOD(year,400) .EQ. 0) then
     	         total_secs = total_secs + secs_in_day
	      else
		 if (MOD(year,4) .EQ. 0 .AND. MOD(year,100) .NE. 0)
     .	             total_secs = total_secs + secs_in_day
	      endif
	   endif
*          ADD SECONDS FOR NUMBER OF DAYS
	   total_secs = total_secs + secs_in_day*(day-1)

*          ADD SECONDS FOR NUMBER OF HOURS
	   total_secs = total_secs + secs_in_hour*(hour)

*          ADD SECONDS FOR NUMBER OF MINUTES
	   total_secs = total_secs + secs_in_minute*(minute)

*          ADD SECONDS FOR NUMBER OF SECS FOR TOTAL NUMBER OF SECONDS
	   total_secs = total_secs + second

******************************************************************************
* ALL OTHER SUPPORTED AND REGULAR CALENDARS
******************************************************************************
	else if (calendar_id .eq. common   .or.
     .           calendar_id .eq. leap_yr  .or. 
     .           calendar_id .eq. days_360 .or. 
     .           calendar_id .eq. tmap_360 .or. 
     .           calendar_id .eq. peculiar) then

*	   INITIALIZE
	   secs_in_year	   = secs_in_day*num_days
	   secs_in_century = 100*secs_in_year

*          ADD LOTS OF SECONDS FOR EACH CENTURY SINCE 0 DAY
	   total_secs = secs_in_century * INT(year/100)

*          ADD A YEAR FOR EACH YEAR SINCE TURN OF CENTURY
	   total_secs = total_secs + secs_in_year*MOD(year,100)

*          ADD SECONDS FOR NUMBER OF MONTHS
	   total_secs = 
	1	total_secs + days_before_month(month)*secs_in_day

*          ADD SECONDS FOR NUMBER OF DAYS
	   total_secs = total_secs + secs_in_day*(day-1)

*          ADD SECONDS FOR NUMBER OF HOURS
	   total_secs = total_secs + secs_in_hour*(hour)

*          ADD SECONDS FOR NUMBER OF MINUTES
	   total_secs = total_secs + secs_in_minute*(minute)

*          ADD SECONDS FOR NUMBER OF SECS FOR TOTAL NUMBER OF SECONDS
	   total_secs = total_secs + second

	else
	   stop 'Illegal calendar in tm_calendars_secs_from_bc()'
	end if

*       FINISHED
	tm_calendars_secs_from_bc = total_secs
	return
	end
