	SUBROUTINE GET_PREC_DIGITS( lo_val, hi_val, small_val, places, right )

* determine the number of characters and the number of decimal places needed
* to represent the range of data from lo_val to hi_val with precision small_val

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/18/87
* revision 0.1 - 7/21/88  - eliminated LOG(0) error in big_10 calculation

* calling argument declarations:
	INTEGER		places, right
	REAL		small_val
	REAL*8		lo_val, hi_val

* internal variable declarations:
	INTEGER		big_10, left
	REAL		big

*  left	 - number of digits needed left of decimal
*  right - number of digits needed right of decimal

* decimal places to resolve small_val
	IF ( small_val .GE. 1.0 ) THEN
	   right = 0
	ELSE
	   right = 1 - INT( LOG10(small_val) )
	ENDIF

* integer places to represent the largest label
	big = MAX( ABS(hi_val), ABS(lo_val) )
	IF ( big .EQ. 0 ) THEN
	   left = 1
	ELSE
	   big_10  = INT( LOG10( 1.01*big ) )
	   IF ( big_10 .GT. 0 ) THEN
	      left = big_10 + 1
	   ELSE
	      left = 1			! 0.xxx
	   ENDIF
	   IF ( lo_val .LT. 0 ) left = left + 1	! minus sign
	ENDIF

* total places
	IF ( right .EQ. 0 ) THEN
	   places = left		! xxxx
	ELSE
	   places = left + 1 + right	! xx.xxx
	ENDIF

	RETURN
      END
