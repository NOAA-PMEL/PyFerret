	SUBROUTINE BINOM_WT( wt, wlen )

* create an array of binomial weights
* by repeated convolutions with the filter [1/2,1/2]

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/6/89
* V301: 2/94 - bug fix if wlen=1 (window of width 1)

* calling argument declarations:
	INTEGER	wlen
	REAL	wt(wlen+1)

* internal variable declarations:
	INTEGER i, k

* zero out the array
	DO 10 i = 1, wlen+1
 10	wt(i) = 0.0

	IF ( wlen .GT. 1 ) THEN
* insert [1/2,1/2]
	   wt(wlen  ) = 0.5
	   wt(wlen-1) = 0.5
	ELSE
	   wt(1) = 1.0
	ENDIF
	
* repeat convolution wlen-2 times
	DO 200 k = 1, wlen-2
	   DO 100 i = 1, wlen
 100	   wt(i) = wt(i)*0.5 + wt(i+1)*0.5
 200	CONTINUE

	RETURN
	END	
