	SUBROUTINE LEFT_JUST( source, dest, tlen )

* left justify a source string into a dest blank-filling the remainder

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written in transportable FORTRAN 77

* revision 0.0 - 7/13/87
* revision 0.1 - 8/21/88 - returned length is measured to last non-blank

* calling argument declarations:
	CHARACTER*(*) source, dest
	INTEGER	tlen

* internal variable declarations:
	INTEGER	TM_LENSTR1, slen, dlen, first

* how long are the strings ?
	slen = LEN( source )
	dlen = LEN( dest   )

* find left-most non-blank
	DO 100 first = 1, slen
	   IF ( source(first:first) .NE. ' ' ) GOTO 200
 100	CONTINUE
	first = slen

* insert left-justified string
 200	dest = source( first:slen )

* measure its length
	tlen = TM_LENSTR1( dest )

	RETURN

	END
