	SUBROUTINE BAD_EDGES_SUB( cx,
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit, bad_src )


* context cx contains the limits of the valid data within array X
* fill the edges of array X with bad-data flags

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/16/89 - from BAD_EDGES
* V300:  3/16/93 - allow for possibility of entire region invalid

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
#endif

* calling argument declarations:
	INTEGER cx, slox, shix, sloy, shiy, sloz, shiz, slot, shit
	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit )
	REAL	bad_src

* internal variable declarations:
	INTEGER	i, j, k, l, i_lo, j_lo, k_lo, l_lo, i_hi, j_hi, k_hi, l_hi

* --- end of introductory code ---

* valid limits
	i_lo = cx_lo_s1(cx)
	j_lo = cx_lo_s2(cx)
	k_lo = cx_lo_s3(cx)
	l_lo = cx_lo_s4(cx)
	i_hi = cx_hi_s1(cx)
	j_hi = cx_hi_s2(cx)
	k_hi = cx_hi_s3(cx)
	l_hi = cx_hi_s4(cx)

* 3/93
* if a low valid limit exceeds a high limit then there must be no valid region
        IF ( i_lo .GT. i_hi
     .  .OR. j_lo .GT. j_hi
     .  .OR. k_lo .GT. k_hi
     .  .OR. l_lo .GT. l_hi ) THEN
* fill entire result with bad flags
	   DO 10 i = slox, shix
	   DO 10 j = sloy, shiy
	   DO 10 k = sloz, shiz
	   DO 10 l = slot, shit
 10	   src(i,j,k,l) = bad_src
           RETURN
        ENDIF

* fill in along X axis
	DO 100 i = slox, i_lo-1
	DO 100 j = j_lo, j_hi
	DO 100 k = k_lo, k_hi
	DO 100 l = l_lo, l_hi
 100	src(i,j,k,l) = bad_src
	DO 110 i = i_hi+1, shix
	DO 110 j = j_lo, j_hi
	DO 110 k = k_lo, k_hi
	DO 110 l = l_lo, l_hi
 110	src(i,j,k,l) = bad_src

* fill in along Y axis
	DO 200 j = sloy, j_lo-1
	DO 200 i = i_lo, i_hi
	DO 200 k = k_lo, k_hi
	DO 200 l = l_lo, l_hi
 200	src(i,j,k,l) = bad_src
	DO 210 j = j_hi+1, shiy
	DO 210 i = i_lo, i_hi
	DO 210 k = k_lo, k_hi
	DO 210 l = l_lo, l_hi
 210	src(i,j,k,l) = bad_src

* fill in along Z axis
	DO 300 k = sloz, k_lo-1
	DO 300 i = i_lo, i_hi
	DO 300 j = j_lo, j_hi
	DO 300 l = l_lo, l_hi
 300	src(i,j,k,l) = bad_src
	DO 310 k = k_hi+1, shiz
	DO 310 i = i_lo, i_hi
	DO 310 j = j_lo, j_hi
	DO 310 l = l_lo, l_hi
 310	src(i,j,k,l) = bad_src

* fill in along T axis
	DO 400 l = slot, l_lo-1
	DO 400 i = i_lo, i_hi
	DO 400 j = j_lo, j_hi
	DO 400 k = k_lo, k_hi
 400	src(i,j,k,l) = bad_src
	DO 410 l = l_hi+1, shit
	DO 410 i = i_lo, i_hi
	DO 410 j = j_lo, j_hi
	DO 410 k = k_lo, k_hi
 410	src(i,j,k,l) = bad_src

	RETURN
	END 
