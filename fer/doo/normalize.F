	SUBROUTINE NORMALIZE( wt, wlen )

* create an array to sum to one

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* for Unix/RISC port 9/91

* calling argument declarations:
	INTEGER	wlen
	REAL	wt(wlen)

* internal variable declarations:
	INTEGER i
        REAL sum

* compute initial normalization
        sum = 0.0
	DO 100 i = 1, wlen
 100	sum = sum + wt(i)


* normalize to unity
        DO 200 i = 1, wlen
 200    wt(i) = wt(i) / sum

	RETURN
	END	
