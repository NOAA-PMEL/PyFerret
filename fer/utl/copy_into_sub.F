	SUBROUTINE COPY_INTO_SUB( cxs,
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit  )

* copy data from source grid into dstination grid using limits of source
* context (i.e. dstination grid is ASSUMED) big enough to hold data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/16/89 - from COPY_INTO

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER cxs,
     .		slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit
	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit ),
     .		dst( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit )

* internal variable declarations:
	INTEGER	i, j, k, l

* --- end of introductory code ---

	DO 100 l = cx_lo_s4(cxs), cx_hi_s4(cxs)
	DO 100 k = cx_lo_s3(cxs), cx_hi_s3(cxs)
	DO 100 j = cx_lo_s2(cxs), cx_hi_s2(cxs)
	DO 100 i = cx_lo_s1(cxs), cx_hi_s1(cxs)

	   dst(i,j,k,l) = src(i,j,k,l)

 100	CONTINUE

	RETURN
	END 
