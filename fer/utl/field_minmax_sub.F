	SUBROUTINE FIELD_MINMAX_SUB ( arr, cx,
     .		slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		bad, lo, hi )

* support routine for FIELD_MINMAX with unsubscripted limits

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 3/97

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER cx, slox, shix, sloy, shiy, sloz, shiz, slot, shit
	REAL    arr( slox:shix,sloy:shiy,sloz:shiz,slot:shit ),
     .		bad, lo, hi

* internal variable declarations:
	INTEGER	i, j, k, l

* --- end of introductory code ---

* initialize
	hi = arbitrary_small_val4
	lo = arbitrary_large_val4

	DO 100 l = cx_lo_s4(cx), cx_hi_s4(cx)
	DO 100 k = cx_lo_s3(cx), cx_hi_s3(cx)
	DO 100 j = cx_lo_s2(cx), cx_hi_s2(cx)
	DO 100 i = cx_lo_s1(cx), cx_hi_s1(cx)

	   IF (arr(i,j,k,l) .NE. bad) THEN
	      IF (arr(i,j,k,l) .LT. lo) lo = arr(i,j,k,l)
	      IF (arr(i,j,k,l) .GT. hi) hi = arr(i,j,k,l)
	   ENDIF
	   
 100	CONTINUE

	RETURN
	END 
