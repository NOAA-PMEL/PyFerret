	SUBROUTINE BAD_EDGES_SUB( cx,
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit, bad_src )


*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
