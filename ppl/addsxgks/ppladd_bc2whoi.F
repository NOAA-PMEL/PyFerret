	CHARACTER*14 FUNCTION BC2WHOI( num_minutes )

* convert minutes since 01-JAN-0000 00:00:00 to a WHOI-formatted of the form
* yymmddhhmm

* Written 9/20/88 By Steve Hankin for PMEL/TMAP

* version 0.0 - based on TM_SECS_TO_DATE by Mark Verschell
* version 0.1 - calls TMAP library routines TM_SECS_TO_DATE and TM_BREAK_DATE
* version 0.2 - 1/19/89: remove century from year for WHOI format
* Mod J Davison 11.22.95 Return century, add seconds, for 4 digit year 

* Argument definition
	INTEGER		year, month, day, hour, minute, second, status
	INTEGER		century
	REAL*8		num_minutes

* Local Definition
	CHARACTER*20	TM_SECS_TO_DATE, date

* convert to date string of the form dd-mmm-yyyy:hh:mm:ss
	date = TM_SECS_TO_DATE( num_minutes * 60.D0 )

* break into individual integer date values
	CALL TM_BREAK_DATE (date, year, month, day,
     .				  hour, minute, second, status)

* kludge for WHOI format limitation - year doesn't contain century

	century = year/100
	year = MOD(year,100)

* Put results into WHOI string
	WRITE (BC2WHOI,3000) year, month, day, hour, minute,
     .	second, century
3000	FORMAT(7I2.2)

	RETURN
	END
