	SUBROUTINE CONVOLVE(	idim, hlen, wt,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx )

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
* convolve the component grid, com, with the weight function, wt
* along axis idim
* note: the component context may not be of adequate size for the full
*	calculation.  Missing data flags will be inserted where computation is
*	impossible
* also: when bad data points are encountered in the component data all
*	result data depending on it are flagged as bad, too

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/2/89
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, hlen, com_mr, com_cx, res_mr, res_cx
	REAL	wt(-hlen:hlen),
     .		com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	INTEGER i, j, k, l, ii, lo_sub(4), hi_sub(4)
	REAL	bad_com, bad_res, comp, sum

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, hi_s1, hi_s2, hi_s3, hi_s4
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4))

* initialize
	bad_com = mr_bad_data( res_mr )
	bad_res = mr_bad_data( com_mr )

* compute limits within which calculation is possible
	DO 10 ii = 1, 4
	   lo_sub(ii) = cx_lo_ss(res_cx,ii)
	   hi_sub(ii) = cx_hi_ss(res_cx,ii)
 10	CONTINUE
	lo_sub(idim) = cx_lo_ss(com_cx,idim) + hlen
	hi_sub(idim) = cx_hi_ss(com_cx,idim) - hlen

* !temp - not optimized for speed yet
* ... though treating each axis separately is to avoid testing inside the loop
* CONVOLVE ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 l = lo_s4, hi_s4
	   DO 110 k = lo_s3, hi_s3
	   DO 110 j = lo_s2, hi_s2
	   DO 110 i = lo_s1, hi_s1
	      sum = 0.0
	      DO 100 ii = -hlen, hlen
	         comp = com(i+ii,j,k,l)	! the only line different w/ each axis
	         IF ( comp .EQ. bad_com ) THEN
	            res(i,j,k,l) = bad_res
	            GOTO 110
	         ELSE
	            sum = sum + comp*wt(ii)
	         ENDIF
 100	      CONTINUE
	      res(i,j,k,l) = sum
 110	   CONTINUE

* CONVOLVE ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = lo_s4, hi_s4
	   DO 210 k = lo_s3, hi_s3
	   DO 210 j = lo_s2, hi_s2
	   DO 210 i = lo_s1, hi_s1
	      sum = 0.0
	      DO 200 ii = -hlen, hlen
	         comp = com(i,j+ii,k,l)	! the only line different w/ each axis
	         IF ( comp .EQ. bad_com ) THEN
	            res(i,j,k,l) = bad_res
	            GOTO 210
	         ELSE
	            sum = sum + comp*wt(ii)
	         ENDIF
 200	      CONTINUE
	      res(i,j,k,l) = sum
 210	   CONTINUE

* CONVOLVE ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = lo_s4, hi_s4
	   DO 310 k = lo_s3, hi_s3
	   DO 310 j = lo_s2, hi_s2
	   DO 310 i = lo_s1, hi_s1
	      sum = 0.0
	      DO 300 ii = -hlen, hlen
	         comp = com(i,j,k+ii,l)	! the only line different w/ each axis
	         IF ( comp .EQ. bad_com ) THEN
	            res(i,j,k,l) = bad_res
	            GOTO 310
	         ELSE
	            sum = sum + comp*wt(ii)
	         ENDIF
 300	      CONTINUE
	      res(i,j,k,l) = sum
 310	   CONTINUE

* CONVOLVE ALONG T AXIS
	ELSE
	   DO 410 l = lo_s4, hi_s4
	   DO 410 k = lo_s3, hi_s3
	   DO 410 j = lo_s2, hi_s2
	   DO 410 i = lo_s1, hi_s1
	      sum = 0.0
	      DO 400 ii = -hlen, hlen
	         comp = com(i,j,k,l+ii)	! the only line different w/ each axis
	         IF ( comp .EQ. bad_com ) THEN
	            res(i,j,k,l) = bad_res
	            GOTO 410
	         ELSE
	            sum = sum + comp*wt(ii)
	         ENDIF
 400	      CONTINUE
	      res(i,j,k,l) = sum
 410	   CONTINUE
	ENDIF

	RETURN
	END	
