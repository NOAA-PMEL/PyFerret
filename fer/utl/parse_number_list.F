	SUBROUTINE PARSE_NUMBER_LIST( string, array, nmax, n, status )

* return an array of floating point values from a list of the form
* "{ 1, 3,5,7, 9}"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500: *sh* 12/98

* calling argument declarations
	INTEGER		nmax, n, status
	REAL		array(nmax)
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		TM_DIGIT
	INTEGER 	comma_pos, last_comma_pos, slen
	CHARACTER	buff*64

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* initialize
	n = 0
	slen	  = LEN( string )
	last_comma_pos = 0

* trim around the enclosing "{}" brackets
	IF ( string(slen:slen) .EQ. '}' ) slen = slen - 1
	IF ( string (1:1) .EQ. '{' ) last_comma_pos = 1
	IF (last_comma_pos .GE. slen) GOTO 5000

* loop through the comma list decoding and returning values
 100	CONTINUE
	IF ( .NOT.TM_DIGIT(string(last_comma_pos+1:slen)) ) GOTO 5000

	comma_pos = INDEX( string(last_comma_pos+1:slen), ',' )
	n = n + 1
	If ( n .GT. nmax ) GOTO 5100   ! should never happen
	IF ( comma_pos .EQ. 0 ) THEN
*  ... last item in list
	   buff = string(last_comma_pos+1:slen)
	   READ ( buff, *, ERR=5000 ) array(n)
	   GOTO 1000
	ELSEIF ( comma_pos .EQ. 1 ) THEN
	   array(n) = bad_val4				! 2 commas together
	ELSE
* ... read another value
	   buff = string(last_comma_pos+1:last_comma_pos+comma_pos-1)
	   IF ( buff .EQ. ' ' ) THEN
	      array(n) = bad_val4
	   ELSE
	      READ ( buff, *, ERR=5000 ) array(n)
	   ENDIF
	ENDIF
	last_comma_pos = last_comma_pos + comma_pos
	GOTO 100

* success
 1000	status = ferr_ok
	RETURN

* syntax error
 5000	CALL ERRMSG( ferr_syntax, status, string(last_comma_pos+1:),
     .							*1000 )
 5100	CALL ERRMSG( ferr_internal, status, string, *1000 )
	END
