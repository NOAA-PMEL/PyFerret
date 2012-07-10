	SUBROUTINE BAD_EDGES_SUB( cx,
     .		src, slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif, bad_src )


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
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER cx, slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif
	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit,sloe:shie,slof:shif )
	REAL	bad_src

* internal variable declarations:
	INTEGER	i, j, k, l, m, n, i_lo, j_lo, k_lo, l_lo, 
     .		m_lo, n_lo, i_hi, j_hi, k_hi, l_hi, m_hi, n_hi

* --- end of introductory code ---

* valid limits
	i_lo = cx_lo_s1(cx)
	j_lo = cx_lo_s2(cx)
	k_lo = cx_lo_s3(cx)
	l_lo = cx_lo_s4(cx)
	m_lo = cx_lo_s5(cx)
	n_lo = cx_lo_s6(cx)
	i_hi = cx_hi_s1(cx)
	j_hi = cx_hi_s2(cx)
	k_hi = cx_hi_s3(cx)
	l_hi = cx_hi_s4(cx)
	m_hi = cx_hi_s5(cx)
	n_hi = cx_hi_s6(cx)

* 3/93
* if a low valid limit exceeds a high limit then there must be no valid region
        IF ( i_lo .GT. i_hi
     .  .OR. j_lo .GT. j_hi
     .  .OR. k_lo .GT. k_hi
     .  .OR. l_lo .GT. l_hi
     .  .OR. m_lo .GT. m_hi
     .  .OR. n_lo .GT. n_hi ) THEN
* fill entire result with bad flags
	   DO 10 i = slox, shix
	   DO 10 j = sloy, shiy
	   DO 10 k = sloz, shiz
	   DO 10 l = slot, shit
	   DO 10 m = sloe, shie
	   DO 10 n = slof, shif
 10	   src(i,j,k,l,m,n) = bad_src
           RETURN
        ENDIF

* fill in along X axis
	DO 100 i = slox, i_lo-1
	DO 100 j = j_lo, j_hi
	DO 100 k = k_lo, k_hi
	DO 100 l = l_lo, l_hi
	DO 100 m = m_lo, m_hi
	DO 100 n = n_lo, n_hi
 100	src(i,j,k,l,m,n) = bad_src
	DO 110 i = i_hi+1, shix
	DO 110 j = j_lo, j_hi
	DO 110 k = k_lo, k_hi
	DO 110 l = l_lo, l_hi
	DO 110 m = m_lo, m_hi
	DO 110 n = n_lo, n_hi
 110	src(i,j,k,l,m,n) = bad_src

* fill in along Y axis
	DO 200 j = sloy, j_lo-1
	DO 200 i = i_lo, i_hi
	DO 200 k = k_lo, k_hi
	DO 200 l = l_lo, l_hi
	DO 200 m = m_lo, m_hi
	DO 200 n = n_lo, n_hi
 200	src(i,j,k,l,m,n) = bad_src
	DO 210 j = j_hi+1, shiy
	DO 210 i = i_lo, i_hi
	DO 210 k = k_lo, k_hi
	DO 210 l = l_lo, l_hi
	DO 210 m = m_lo, m_hi
	DO 210 n = n_lo, n_hi
 210	src(i,j,k,l,m,n) = bad_src

* fill in along Z axis
	DO 300 k = sloz, k_lo-1
	DO 300 i = i_lo, i_hi
	DO 300 j = j_lo, j_hi
	DO 300 l = l_lo, l_hi
	DO 300 m = m_lo, m_hi
	DO 300 n = n_lo, n_hi
 300	src(i,j,k,l,m,n) = bad_src
	DO 310 k = k_hi+1, shiz
	DO 310 i = i_lo, i_hi
	DO 310 j = j_lo, j_hi
	DO 310 l = l_lo, l_hi
	DO 310 m = m_lo, m_hi
	DO 310 n = n_lo, n_hi
 310	src(i,j,k,l,m,n) = bad_src

* fill in along T axis
	DO 400 l = slot, l_lo-1
	DO 400 i = i_lo, i_hi
	DO 400 j = j_lo, j_hi
	DO 400 k = k_lo, k_hi
	DO 400 m = m_lo, m_hi
	DO 400 n = n_lo, n_hi
 400	src(i,j,k,l,m,n) = bad_src
	DO 410 l = l_hi+1, shit
	DO 410 i = i_lo, i_hi
	DO 410 j = j_lo, j_hi
	DO 410 k = k_lo, k_hi
	DO 410 m = m_lo, m_hi
	DO 410 n = n_lo, n_hi
 410	src(i,j,k,l,m,n) = bad_src

* fill in along E axis
	DO 500 m = sloe, m_lo-1
	DO 500 i = i_lo, i_hi
	DO 500 j = j_lo, j_hi
	DO 500 l = l_lo, l_hi
	DO 500 k = k_lo, k_hi
	DO 500 n = n_lo, n_hi
 500	src(i,j,k,l,m,n) = bad_src
	DO 510 m = m_hi+1, shie
	DO 510 i = i_lo, i_hi
	DO 510 j = j_lo, j_hi
	DO 510 k = k_lo, k_hi
	DO 510 l = l_lo, l_hi
	DO 510 n = n_lo, n_hi
 510	src(i,j,k,l,m,n) = bad_src

* fill in along F axis
	DO 600 n = slof, n_lo-1
	DO 600 i = i_lo, i_hi
	DO 600 j = j_lo, j_hi
	DO 600 k = k_lo, k_hi
	DO 600 l = l_lo, l_hi
	DO 600 m = m_lo, m_hi
 600	src(i,j,k,l,m,n) = bad_src
	DO 610 n = n_hi+1, shif
	DO 610 i = i_lo, i_hi
	DO 610 j = j_lo, j_hi
	DO 610 k = k_lo, k_hi
	DO 610 l = l_lo, l_hi
	DO 610 m = m_lo, m_hi
 610	src(i,j,k,l,m,n) = bad_src

	RETURN
	END 
