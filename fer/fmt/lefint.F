	CHARACTER*(*) FUNCTION LEFINT( ival, llen )

* create a left-justified integer string from ival
* typical usage:
* i) declare length of function in calling routine 
*	eg. CHARACTER*8 LEFINT
* ii) use as a character variable
*	eg. WRITE( 20,'(1X,A)') LEFINT( j )

* note: no error checking is done to see that ival will fit into the declared
*	field width of LEFINT.  The integer will be truncated on the right
*	if the field is too small resulting in erroneous output.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written in transportable FORTRAN 77

* revision 0.0 - 6/30/87
* revision 0.1 - 9/28/87 - added llen

* calling argument declarations:
	INTEGER	ival, llen

* internal variable declarations:
	CHARACTER*16 buff16
	INTEGER	     first

* create right-justified string of up to 16 digits
	WRITE ( buff16, '(I16)' ) ival

* find left-most non-blank
	DO 100 first = 1, 15
	   IF ( buff16(first:first) .NE. ' ' ) GOTO 200
 100	CONTINUE
	first = 16

* insert left-justified string
 200	LEFINT = buff16(first:16)
	llen   = 17 - first
	RETURN
	END
