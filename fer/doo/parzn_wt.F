	SUBROUTINE PARZN_WT( wt, wlen )

* create an array of Parzen filter (window) weights

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* for Unix/RISC port 9/91 - based on BINOM_WT

* calling argument declarations:
	INTEGER	wlen
	REAL	wt(0:wlen-1)

* internal variable declarations:
	INTEGER i
        REAL c1, c2

* based on Numerical Recipes, W.H.Press, et. al. p.425
        c1 = (wlen + 1) / 2.
        c2 = (wlen - 1) / 2.
	DO 100 i = 0, wlen-1
 100	wt(i) = 1.0 - ABS( (i-c2)/c1 )

* normalize to 1
        CALL NORMALIZE( wt, wlen )

	RETURN
	END	
