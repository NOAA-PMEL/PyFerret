	REAL*8 FUNCTION TM_SECS_FROM_BC ( year, month, day,
     .					   hour, minute, second)
* Will convert a date of yyyy-mm-dd hh:mm:ss to seconds since
* 0000-01-01 00:00:00
*
* Written 6/16/87 By Mark Verschell for PMEL/TMAP
*
* revision 0.00 - 06/16/87 - Initial Incarnation
*
* Argument definition
	INTEGER		year, month, day, hour, minute,second
*
* Local Definition
	INTEGER		days_in_month(12)
	REAL*8		secs_in_minute, secs_in_hour, secs_in_day,
     .			secs_in_year, secs_in_century, total_secs

	PARAMETER	(secs_in_minute	 = 60.,
     .			 secs_in_hour	 = secs_in_minute*60.,
     .			 secs_in_day	 = secs_in_hour*24.,
     .			 secs_in_year	 = secs_in_day*365.,
     .			 secs_in_century = secs_in_day*36524.)

	DATA days_in_month /   0,  31,  59,  90, 120, 151,
     .			     181, 212, 243, 273, 304, 334/

* add lots of seconds for each century since 0 day
	total_secs = secs_in_century * INT(year/100)
* add a day for every 400 years (leap year at centennial)
	total_secs = total_secs + secs_in_day*INT(year/400.+0.9975)
* add a year for each year since turn of century
	total_secs = total_secs + secs_in_year*MOD(year,100)
* add a day for each leap year since centennial (except for centennial)
	total_secs = total_secs + secs_in_day*INT((MOD(year,100)-1)/4)
* add seconds for number of months
	total_secs = total_secs + days_in_month(month)*secs_in_day
* add 1 day worth of seconds if this is leap year and past february
	IF (month .GT. 2) THEN
	  IF (MOD(year,400) .EQ. 0) THEN
	    total_secs = total_secs + secs_in_day
	  ELSE
	    IF (MOD(year,4) .EQ. 0 .AND. MOD(year,100) .NE. 0)
     .	          total_secs = total_secs + secs_in_day
	  ENDIF
	ENDIF
* add seconds for number of days
	total_secs = total_secs + secs_in_day*(day-1)
* add seconds for number of hours
	total_secs = total_secs + secs_in_hour*(hour)
* add seconds for number of minutes
	total_secs = total_secs + secs_in_minute*(minute)
* add seconds for number of secs for total number of seconds
	total_secs = total_secs + second

* finished
	TM_SECS_FROM_BC = total_secs
	RETURN
	END
