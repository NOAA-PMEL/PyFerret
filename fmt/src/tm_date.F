	LOGICAL FUNCTION TM_DATE( string )

* naive test to see if the given string may be a date
* complete date format is dd-mmm-yy:hh:mm:ss where mmm is a string
* naive test insists that
*     1) first character is a digit
*     2) contains : or -
*     3) if "-" must be followed by 3 characters

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 3/12/91 needed for Unix/RISC port
*
* calling argument declarations:
	CHARACTER*(*)	string

* internal variable declarations:
	INTEGER		i, ip, slen
	CHARACTER*1	c
	
* local parameter declarations:
        CHARACTER*1     tab
        PARAMETER     ( tab = CHAR(9) )

* initialize assuming failure
	TM_DATE = .FALSE.

* skip leading blanks
        slen = LEN(string)
	DO 100 i = 1, slen
	   c = string(i:i)
	   IF ( c .NE. ' ' .AND. c .NE. tab ) GOTO 200
 100	CONTINUE
* string is blank
	RETURN

* is first character a digit ?
* modified 6/92 -- kob
* 200    IF ( c .LE. '0' .OR. c .GE. '9' ) RETURN
 200    IF ( c .LT. '0' .OR. c .GT. '9' ) RETURN

* is there a colon ?
        IF ( INDEX( string, ':') .GT. 0 ) THEN
           TM_DATE = .TRUE.
           RETURN
        ENDIF

* is there a dash ?
        ip = INDEX( string, '-')
        IF ( ip .EQ. 0 ) RETURN

* do three letters follow the dash ?
        IF ( slen .LT. ip+3 ) RETURN
        DO 300 i = ip+1, ip+3
           c = string(i:i)
           IF ( ( c.GE.'A' .AND. c.LE.'Z' )
     .    .OR.  ( c.GE.'a' .AND. c.LE.'z' ) ) GOTO 300
* not a letter
	   RETURN
 300    CONTINUE

* it meets the requirements
        TM_DATE = .TRUE.
        RETURN

        END
