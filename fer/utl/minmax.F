	SUBROUTINE MINMAX( array, n, bad_flag, lo, hi )

* determine the minimum and maximum values within a REAL*4 array

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/20/88

#ifdef unix
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	n
	REAL    array(n), bad_flag, lo, hi

* internal variable declarations:
	INTEGER	i

* initialize
	lo = arbitrary_large_val4
	hi = arbitrary_small_val4

* search
	DO 100 i = 1, n

	   IF ( array(i) .EQ. bad_flag ) GOTO 100
	   IF ( array(i) .LT. lo ) lo = array(i)
	   IF ( array(i) .GT. hi ) hi = array(i)

 100	CONTINUE

	RETURN
	END
