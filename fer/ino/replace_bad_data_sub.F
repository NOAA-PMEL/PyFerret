	SUBROUTINE REPLACE_BAD_DATA_SUB( cx, old_bad,
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit, bad_src )

* replace all occurances of the value "old_bad" with the bad data flag
* associated with this mr

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/19/87
* V200:  10/16/89 - based on REPLACE_BAD_DATA

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, slox, shix, sloy, shiy, sloz, shiz, slot, shit
	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit )
	REAL	bad_src, old_bad

* internal variable declarations:
	INTEGER i, j, k, l

* replace the old flags
	DO 100 l = slot, shit
	DO 100 k = sloz, shiz
	DO 100 j = sloy, shiy
	DO 100 i = slox, shix
	   IF ( src(i,j,k,l) .EQ. old_bad ) src(i,j,k,l) = bad_src
 100	CONTINUE

	RETURN
	END

