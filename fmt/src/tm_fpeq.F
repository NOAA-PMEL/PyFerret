	LOGICAL FUNCTION TM_FPEQ ( a, b )

* test to see if REAL*4 floating point numbers are equal to within
* "reasonable" accuracy
* REAL*4 (F_floating) machine error on the VAX is "approximately one part in
* 2**23" or 1.2E-7.  Allow 4.E-7 as machine error after many typical roundoffs.
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 11/21/85
* revision 0.10 - 10/23/87 - all variables explicitly declared
*
* calling arguments:
	REAL	a, b

* local parameters
	REAL	   VAX_epsilon
	PARAMETER (VAX_epsilon = 4.E-7)

* treat relative error case (non-zero) separate fron zero case

	IF  (a.NE.0.0   .AND.  b.NE.0.0 )  THEN

	   tm_fpeq =  ABS(a-b) .LE. ABS(b)*VAX_epsilon		! relative error

	ELSE

	   tm_fpeq =	ABS(a) .LT. VAX_epsilon
     .		.AND.   ABS(b) .LT. VAX_epsilon			! absolute error

	ENDIF
	RETURN
	END
