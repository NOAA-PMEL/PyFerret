	LOGICAL FUNCTION TM_DIGIT( string )

* test to see if the first character of "string" is the first character
* of a number

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 06/24/88
* 11/6/94 *sh* Kludge for AIX port (fixes to PLOT+ to avoid problems with
*	FORTRAN READ ERR= branches):  check for "/" following "." since
*	path names often begin with "./name"

* calling argument declarations:
	CHARACTER*(*)	string

* internal variable declarations:
	INTEGER		i
	CHARACTER*1	c
	
* local parameter declarations:
        CHARACTER*1     tab
        PARAMETER     ( tab = CHAR(9) )

* skip leading blanks
	DO 100 i = 1, LEN(string)
	   c = string(i:i)
	   IF ( c .NE. ' ' .AND. c .NE. tab ) GOTO 200
 100	CONTINUE

* string is blank
	TM_DIGIT = .FALSE.
	
* check first character
 200	TM_DIGIT =   c .EQ. '+'
     .		.OR. c .EQ. '-'
     .		.OR. c .EQ. '.'
     .		.OR. c .GE. '0' .AND. c .LE. '9'

* kludge check for "./name" (11/94 for AIX port)
	IF (c.EQ.'.' .AND. LEN(string).GE.2 )
     .		TM_DIGIT = string(2:2) .NE. '/'

	RETURN

      END
