	SUBROUTINE EQUAL_STRING( string, val, status )

* from a string of the form "name=value", return the "value" string
* if the string is simply "name" then return ' ' (blank)
* change the string to upper case

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/27/88 - based on EQUAL_VAL
* V200:   7/7/89 -  remove leading blanks, process quotation marks
*	  9/22/89 - fixed bug for NAME="" (close quotes)

* calling argument declarations
	INTEGER		status
	CHARACTER*(*)	string, val

* internal variable declarations
	INTEGER	TM_LENSTR1, STR_UPCASE, equal_pos, slen, strt, vax_code

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* initialize
	slen	  = TM_LENSTR1( string )
	equal_pos = INDEX( string, '=' )

	IF ( equal_pos .EQ. 0 ) THEN
	   val = ' '				! "name", alone

	ELSEIF ( equal_pos .EQ. slen ) THEN	
	   GOTO 5000				! "name ="

	ELSE

* find first non-blank
	   DO 100 strt = equal_pos+1, slen
 100	   IF ( string(strt:strt) .NE. ' ' ) GOTO 110
	   STOP '=_str'		! cannot occur

* remove quotation marks
 110	   IF ( string(strt:strt) .EQ. '"' ) strt = strt + 1
	   IF ( string(slen:slen) .EQ. '"' ) slen = slen - 1

	   IF ( strt .GT. slen ) THEN
	      val = ' '
	   ELSE
	     vax_code = STR_UPCASE( val, string(strt:slen) )
	   ENDIF

	ENDIF

* success
	status = ferr_ok
 1000	RETURN

* syntax error
 5000	CALL ERRMSG( ferr_syntax, status, string, *1000 )
	END
