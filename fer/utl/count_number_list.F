	SUBROUTINE COUNT_NUMBER_LIST( string, n )

* count the number of values in a list of the form "{ 1, 3,5,7, 9}"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500: *sh* 12/98
* V510: *sh* 10/99 bug fix: single character {3} gave n=0

* calling argument declarations
	INTEGER		n
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		no_digits_yet
	INTEGER 	i, start, send
	CHARACTER*1	c

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* initialize
	n = 0
	send = LEN( string )
	start = 1
	no_digits_yet = .TRUE.

* trim around the enclosing "{}" brackets
	IF ( string(send:send) .EQ. '}' ) send = send - 1
	IF ( string (1:1) .EQ. '{' ) start = start + 1
	IF (start .GT. send) RETURN 

* loop through the text counting commas
	DO 100 i = start, send
	   c = string(i:i)
	   IF (c .EQ. ',') n = n+1
	   IF (no_digits_yet) THEN
	      IF (c.GE.'0' .AND. c.LE.'9') no_digits_yet = .FALSE.
	   ENDIF
 100	CONTINUE

* if there were any digits at all then the count of numbers is the
* count of commas plus 1
	IF (.NOT.no_digits_yet) n = n + 1

* success
	RETURN

	END
