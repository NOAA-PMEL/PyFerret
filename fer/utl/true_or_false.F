	LOGICAL FUNCTION TRUE_OR_FALSE (string, status)

* check to see if the string represents a TRUE condition or a FALSE one
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420  4/25/96
*	7/96 - minor compiler bug fix -- eliminate unuser RETURN

* calling argument declarations:
	INTEGER		status
	CHARACTER*(*)	string

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* local parameter declarations
	INTEGER nyes, nno
	PARAMETER (nyes = 4,
     .		   nno  = 6)

* local variable declarations
	LOGICAL TM_DIGIT
	INTEGER	STR_MATCH, i
	REAL	value
	CHARACTER*8 yes_strings(nyes), no_strings(nno)

	DATA (yes_strings(i),i=1,nyes)/'TRUE','T','Y','YES'/
	DATA (no_strings(i),i=1,nno)/'FALSE','F','N','NO','BAD','MISSING'/
	
* initialize
	status = ferr_ok

* a blank string is a "false"
	IF ( string .EQ. ' ' ) GOTO 2000

* if the string is numerical then "0" is false, all other is true
	IF ( TM_DIGIT(string) ) THEN
	   READ ( string, *, ERR=5100) value
	   IF ( value .EQ. 0.0 ) THEN
	      GOTO 2000
	   ELSE
	      GOTO 1000
	   ENDIF

* if the string is non-numerical then try matching
	ELSE
	   i = STR_MATCH( string, yes_strings, nyes )
	   IF ( i .NE. atom_not_found ) GOTO 1000	! a valid TRUE
	   i = STR_MATCH( string, no_strings,  nno  )
	   IF ( i .NE. atom_not_found ) GOTO 2000	! a valid FALSE
	   GOTO 5100					! syntax err

	ENDIF

* return TRUE
 1000	TRUE_OR_FALSE = .TRUE.
	RETURN

* return FALSE
 2000	TRUE_OR_FALSE = .FALSE.
	RETURN


* error exit
 5100	TRUE_OR_FALSE = .FALSE.		! included for whiny OSF compiler
        CALL ERRMSG( ferr_syntax, status, string, *2000 )
	END
