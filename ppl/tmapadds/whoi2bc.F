	REAL*8 FUNCTION WHOI2BC( whoi )

* Convert a WHOI-formatted date of yymmddhhmm to minutes since 1-JAN-0000
*
* Steve Hankin - NOAA/PMEL/TMAP
*
* version 0.0 - based on TM_SECS_FROM_BC by Mark Verschell
* version 0.1 - calls TM_SECS_FROM_BC
* version 0.2 - bug fix: add 1900 to year
* Mod 11.22.95 J Davison for 4 digit year 
*
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

* Argument definition
	CHARACTER*(*)	whoi
*
* Local Definition
	INTEGER		year, month, day, hour, minute, second
	INTEGER		century
	REAL*8		TM_SECS_FROM_BC

* read the WHOI-formatted string
	READ ( whoi, 3000 ) year, month, day, hour, minute, second, century
 3000	FORMAT (7I2.2)
!!!!!	second = 0
	year = year + century*100

* call TMAP library routine and convert seconds to minutes
	WHOI2BC = TM_SECS_FROM_BC( year, month, day, hour, minute, second )
     .		  / 60.D0

	RETURN
	END
