	SUBROUTINE EQUAL_VAL( string, val, status )

* from a string of the form "name=value", return the value
* if the string is simply "name" then return unspecified value

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/14/88
* revision 0.1 -  9/8/88 - explicit check for digit so words beginning with T
*			   aren't treated as .TRUE. by VAX Fortran 
* V420: 10/95 - need to buffer "READ(*)" for AIX bort

* calling argument declarations
	INTEGER		status
	REAL		val
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		TM_DIGIT
	INTEGER 	equal_pos, slen
#ifdef AIX_XLF
	CHARACTER	buff*64
#endif

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* initialize
	slen	  = LEN( string )
	equal_pos = INDEX( string, '=' )

	IF ( equal_pos .EQ. 0 ) THEN
	   val = unspecified_val4		! "name", alone

	ELSEIF ( equal_pos .EQ. slen ) THEN	
	   GOTO 5000				! "name ="

	ELSEIF ( string(equal_pos+1:) .EQ. ' ' ) THEN
	   GOTO 5000				! "name =    "

	ELSEIF ( .NOT.TM_DIGIT(string(equal_pos+1:)) ) THEN
	   GOTO 5000				! "name = text"

	ELSE
#ifdef AIX_XLF
	   buff = string(equal_pos+1:)
	   READ ( buff, *, ERR=5000 ) val
#else
	   READ ( string(equal_pos+1:), *, ERR=5000 ) val
#endif
	ENDIF

* success
	status = ferr_ok
 1000	RETURN

* syntax error
 5000	CALL ERRMSG( ferr_syntax, status, string, *1000 )
	END
