	SUBROUTINE EXTRACT_LINE_SUB( cx, line,
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit )

* extract a single line of data from a memory variable grid.  Determine
* which line to extract from the given context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  10/16/89 - based on EXTRACT_LINE

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, slox, shix, sloy, shiy, sloz, shiz, slot, shit
	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit ),
     .		line(*)

* internal variable declarations:
	INTEGER	i, j, k, l, iout

* extract the line
	iout = 0
	DO 200 l = cx_lo_s4(cx), cx_hi_s4(cx)
	DO 200 k = cx_lo_s3(cx), cx_hi_s3(cx)
	DO 200 j = cx_lo_s2(cx), cx_hi_s2(cx)
	DO 200 i = cx_lo_s1(cx), cx_hi_s1(cx)

	   iout = iout + 1
	   line( iout ) = src( i,j,k,l )

 200	CONTINUE

	RETURN
	END
