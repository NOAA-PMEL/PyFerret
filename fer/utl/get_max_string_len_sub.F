	SUBROUTINE GET_MAX_STRING_LEN_SUB(cx, grid_data, 
     .		lox, hix, loy, hiy, loz, hiz, lot, hit, 
     .		loe, hie, lof, hif, smax )

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
* Return the length of the longest string in this context

* V530: 8/00 *sh*
* v554: 12/04 *acm* Get the shape of the region for the context of the data;
*                   these are the indices to use in looping through grid_data.
*                   (fixes bug 617, listing string and numeric data together)
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
        include 'ferret.parm'
	include 'xcontext.cmn'

* calling argument declarations
	INTEGER  cx, smax, lox, hix, loy, hiy, loz, hiz, 
     .		 lot, hit, loe, hie, lof, hif
	REAL*8 grid_data( lox:hix,loy:hiy,loz:hiz,lot:hit,loe:hie,lof:hif )

* internal variable declarations:
	INTEGER  GET_C_STRING_LEN, GET_ONED_STRING_LEN_SUB, 
     .         i, j, k, l, m, n, ii, slen, 
     .         sslo(nferdims), sshi(nferdims),
     .         ndim, odim, lo, hi, cxperm(nferdims)

* initialize
	smax = 1

        CALL get_cx_dims (cx,ndim,cxperm)

        sslo(1) = cx_lo_s1(cx)
        sslo(2) = cx_lo_s2(cx)
        sslo(3) = cx_lo_s3(cx)
        sslo(4) = cx_lo_s4(cx)
        sslo(5) = cx_lo_s5(cx)
        sslo(6) = cx_lo_s6(cx)

        sshi(1) = cx_hi_s1(cx)
        sshi(2) = cx_hi_s2(cx)
        sshi(3) = cx_hi_s3(cx)
        sshi(4) = cx_hi_s4(cx)
        sshi(5) = cx_hi_s5(cx)
        sshi(6) = cx_hi_s6(cx)


* If it's 1-D the data may not have been permuted.  Just treat the 
* data as a true 1-D set of strings

	IF (ndim .EQ. 1) THEN
	ii = unspecified_int4
	   DO i = 1, nferdims
	      IF (sshi(i)-sslo(i) .GT. 0) THEN
	         lo = sslo(i)
	         hi = sshi(i)
	         ii = lo
	      ENDIF
	   ENDDO
	   IF (ii .EQ. unspecified_int4) THEN
	   DO i = 1, nferdims
	      IF (sshi(i) .NE. unspecified_int4) THEN
	         lo = sslo(i)
	         hi = sshi(i)
	         ii = lo
	      ENDIF
	   ENDDO
	   ENDIF
	ENDIF

	DO n = sslo(cxperm(6)), sshi(cxperm(6))
	DO m = sslo(cxperm(5)), sshi(cxperm(5))
	DO l = sslo(cxperm(4)), sshi(cxperm(4))
	DO k = sslo(cxperm(3)), sshi(cxperm(3))
	DO j = sslo(cxperm(2)), sshi(cxperm(2))
	DO i = sslo(cxperm(1)), sshi(cxperm(1))

	   IF (ndim.EQ.1) THEN
	      slen = GET_ONED_STRING_LEN_SUB( grid_data, lo,hi, ii )
	      ii = ii+1
	   ELSE
	      slen = GET_C_STRING_LEN( grid_data(i,j,k,l,m,n) )
	   ENDIF
	   IF ( slen .GT. smax ) smax = slen

	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

	RETURN
	END
 
