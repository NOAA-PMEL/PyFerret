	CHARACTER*(*) FUNCTION LEFT_REAL( val, fmt, len )

* create a left-justified real string from val with format fmt
* typical usage:
* i) declare length of function in calling routine 
*	eg. CHARACTER*8 LEFT_REAL
* ii) use as a character variable
*	eg. WRITE( 20,'(1X,A)') LEFT_REAL( x, 2, len )

* note: if value cannot fit within field '****' will be inserted

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written in transportable FORTRAN 77

* revision 0.0 - 7/13/87 modeled after LEFINT rev. 0.0

* calling argument declarations:
	REAL	val
	CHARACTER*(*) fmt
	INTEGER	len

* internal variable declarations:
	CHARACTER*16 buff16
	INTEGER	     first

* create right-justified string of up to 16 digits
	WRITE ( buff16, fmt ) val

* find left-most non-blank
	DO 100 first = 1, 15
	   IF ( buff16(first:first) .NE. ' ' ) GOTO 200
 100	CONTINUE
	first = 16

* insert left-justified string
 200	LEFT_REAL = buff16(first:16)

* measure its length
	DO 300 len = 0, 15
	   IF ( LEFT_REAL( len+1:len+1 ) .EQ. ' ' ) RETURN
 300	CONTINUE

* error - didn't fit
	LEFT_REAL = '****'
	len = 4
	RETURN

	END
