	CHARACTER*(*) FUNCTION PAREN_SURROUND( string, slen )

* return the given string with enclosing parens around it and its length
* if the given string is blank then return a blank string

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/11/88

* calling argument declarations
	INTEGER		slen
	CHARACTER*(*)	string

* internal variable declarations
	INTEGER		TM_LENSTR1

	IF ( string .EQ. ' ' ) THEN
	   PAREN_SURROUND = ' '
	   slen = 0

	ELSE
	   slen = TM_LENSTR1( string )
	   PAREN_SURROUND = '(' // string(:slen) // ')'
	   slen = slen + 2

	ENDIF
	RETURN
	END
