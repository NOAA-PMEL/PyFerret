	SUBROUTINE HANNG_WT( wt, wlen )

* create an array of Hanning filter weights

* programmer - steve hankin and Paul Rogers
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* for Unix/RISC port 8/91 - based on BINOM_WT

* calling argument declarations:
	INTEGER	wlen
	REAL	wt(wlen)

* internal variable declarations:
	INTEGER i, use_len
        REAL c

* based on Numerical Recipes, W.H.Press, et. al. p.425
* note, however, that the algorithm given includes a zero point at each end
* of the Hanning filter -> i.e. the number of non-zero weight points is N-2
* What is provided by FERRET will technically be a Hanning window of length N+2
        use_len = wlen + 2
        c = 2. * 3.1415926 / (use_len - 1)
	DO 100 i = 1, wlen
 100	wt(i) = 0.5 * ( 1.0 - COS(c*i) )

* normalize to 1
        CALL NORMALIZE( wt, wlen )
 
	RETURN
	END	
