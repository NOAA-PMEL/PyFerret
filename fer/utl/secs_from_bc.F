	REAL*8 FUNCTION SECS_FROM_BC( date, status )

* return a REAL*8 number giving the number of seconds since 1-JAN-0000:00:00

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 4/7/88
* revision 0.1 1/18/89 - BREAK_DATE --> TM_BREAK_DATE
* V300: 2/93 - return **something** even if error exit (caught by f77 3.2)

* Calling argument declarations:
	INTEGER		status
	CHARACTER*(*)	date

* Local variable declarations:
	INTEGER		year, month, day, hour, minute, second
	REAL*8		TM_SECS_FROM_BC

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* Break up the date string to it's various components
	CALL TM_BREAK_DATE ( date,
     .			  year, month, day, hour, minute, second,
     .			  status )
	IF ( status .NE. ferr_ok ) THEN
           SECS_FROM_BC = -999
           RETURN               ! err not reported
        ENDIF

* Find the offset from 01-JAN-0000 00:00:00 for this date
	SECS_FROM_BC = TM_SECS_FROM_BC (year, month, day,
     .				     hour, minute, second)

	RETURN
	END
