	CHARACTER*(*) FUNCTION LEGAL_CHARS( string, leg_len )

* remove all non-character (non-digit) data from a string - return final length

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/26/87

* calling argument declarations
	CHARACTER*(*)	string
	INTEGER		leg_len

* local variable declarations:
	INTEGER TM_LENSTR, mlen, slen, s
	CHARACTER char1*1

* initialize
	slen	= TM_LENSTR( string )
	mlen	= LEN( LEGAL_CHARS )
	leg_len	= 0

* loop through input string character by character
	DO 100 s = 1, slen
	   char1 = string(s:s)
	   IF ( char1 .GE. '0' .AND. char1 .LE. '9'
     .	   .OR. char1 .GE. 'A' .AND. char1 .LE. 'Z'
     .	   .OR. char1 .GE. 'a' .AND. char1 .LE. 'z' ) THEN
	      leg_len = leg_len + 1
	      LEGAL_CHARS( leg_len:leg_len ) = char1
	      IF ( leg_len .EQ. mlen ) RETURN
	   ENDIF
 100	CONTINUE

	RETURN
	END
