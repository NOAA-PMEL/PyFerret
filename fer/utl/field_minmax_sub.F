	SUBROUTINE FIELD_MINMAX_SUB ( arr, cx,
     .		slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		sloe, shie, slof, shif, bad, lo, hi )

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
* support routine for FIELD_MINMAX with unsubscripted limits

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 3/97
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include 'ferret.parm'
	include	'xcontext.cmn'

* calling argument declarations:
	INTEGER cx, slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif
	REAL    arr( slox:shix,sloy:shiy,sloz:shiz,slot:shit,sloe:shie,slof:shif ),
     .		bad, lo, hi

* internal variable declarations:
	INTEGER	i, j, k, l, m, n

* --- end of introductory code ---

* initialize
	hi = arbitrary_small_val4
	lo = arbitrary_large_val4

	DO n = cx_lo_s6(cx), cx_hi_s6(cx)
	DO m = cx_lo_s5(cx), cx_hi_s5(cx)
	DO l = cx_lo_s4(cx), cx_hi_s4(cx)
	DO k = cx_lo_s3(cx), cx_hi_s3(cx)
	DO j = cx_lo_s2(cx), cx_hi_s2(cx)
	DO i = cx_lo_s1(cx), cx_hi_s1(cx)

	   IF (arr(i,j,k,l,m,n) .NE. bad) THEN
	      IF (arr(i,j,k,l,m,n) .LT. lo) lo = arr(i,j,k,l,m,n)
	      IF (arr(i,j,k,l,m,n) .GT. hi) hi = arr(i,j,k,l,m,n)
	   ENDIF

	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

	RETURN
	END 
