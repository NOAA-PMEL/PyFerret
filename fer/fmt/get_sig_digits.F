	SUBROUTINE GET_SIG_DIGITS ( value, nsig_digits, nleft, nright )

* determine how many digits are needed left and right of decimal point to
* represent "value" with nsig_digits significant digits

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 5/2/86

* calling argument declarations:
	INTEGER	nsig_digits, nleft, nright
	REAL	value

* internal variable declarations:
	REAL	point
	INTEGER	ipoint

	IF ( value .EQ. 0.0 ) THEN
	   point = -0.1					! like value = 0.1
	ELSE
	   point  = LOG10( ABS(value) )
	ENDIF
	IF ( point .LT. 0.0 ) point = point - 1.0	! to round down
	ipoint = INT( point ) + 1

	nleft  = MAX( 0, ipoint )
	nright = MAX( 0, nsig_digits-ipoint )

	RETURN
	END
