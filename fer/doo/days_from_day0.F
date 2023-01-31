	REAL FUNCTION DAYS_FROM_DAY0( date0_days, year, month, day,
     .       rdum, status)


*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* return a relative day number since the day of date0_days where date0_days
* is in units of days since 1-JAN-0000

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V320 11/94 - in order to implement the DAY1900 function
*		based on tm_secs_from_bc by Mark Verschell
* V530 12/00 *sh* - bug fix: avoid crash if month is not in [1,12]
* V600 4/06  *acm*  return the result in rdum as well, for 64-bit build
* V701 *acm* 7/16  ticket 2449: report incorrect choice of date/ eurodate as an error

        IMPLICIT NONE
        include 'errmsg.parm'

* calling argument declarations:

	REAL*8 date0_days
	INTEGER year, month, day, status
        REAL rdum

* Local Definition
	INTEGER		days_in_month(12), mnth
	REAL*8		secs_in_day,
     .			secs_in_year, secs_in_century, total_secs,
     .			total_days

	PARAMETER	(secs_in_day	 = 60.*60.*24.,
     .			 secs_in_year	 = secs_in_day*365.,
     .			 secs_in_century = secs_in_day*36524.)

	DATA days_in_month /   0,  31,  59,  90, 120, 151,
     .			     181, 212, 243, 273, 304, 334/

* to avoid crash for out of bounds month
	IF (month .GT. 12) THEN
	   mnth = 12
	   status = ferr_out_of_range 
	ELSEIF (month .LT. 1) THEN
	   mnth = 1
	   status = ferr_out_of_range 
	ELSE
	   mnth = month
	ENDIF

* add lots of seconds for each century since 0 day
	total_secs = secs_in_century * INT(year/100)
* add a day for every 400 years (leap year at centennial)
	total_secs = total_secs + secs_in_day*INT(year/400.+0.9975)
* add a year for each year since turn of century
	total_secs = total_secs + secs_in_year*MOD(year,100)
* add a day for each leap year since centennial (except for centennial)
	total_secs = total_secs + secs_in_day*INT((MOD(year,100)-1)/4)
* add seconds for number of months
	total_secs = total_secs + days_in_month(mnth)*secs_in_day
* add 1 day worth of seconds if this is leap year and past february
	IF (mnth .GT. 2) THEN
	  IF (MOD(year,400) .EQ. 0) THEN
	    total_secs = total_secs + secs_in_day
	  ELSE
	    IF (MOD(year,4) .EQ. 0 .AND. MOD(year,100) .NE. 0)
     .	          total_secs = total_secs + secs_in_day
	  ENDIF
	ENDIF
* add seconds for number of days
	total_secs = total_secs + secs_in_day*(day-1)

* convert to days and return result
	total_days = total_secs/secs_in_day - date0_days
	DAYS_FROM_DAY0 = total_days

        rdum = DAYS_FROM_DAY0

        RETURN
	END

