	SUBROUTINE EQUAL_STR_LC( string, val, status )

* from a string of the form "name=value", return the "value" string
* if the string is simply "name" then return ' ' (blank)
* this version preserves the upper/lower case of the string

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: revision 0.0 - 3/12/91 - for Unix/RISC port - based on EQUAL_SRING

* calling argument declarations
	INTEGER		status
	CHARACTER*(*)	string, val

* internal variable declarations
	INTEGER	TM_LENSTR1, equal_pos, slen, strt

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
              val = string(strt:slen)
	   ENDIF

	ENDIF

* success
	status = ferr_ok
 1000	RETURN

* syntax error
 5000	CALL ERRMSG( ferr_syntax, status, string, *1000 )
	END
