	SUBROUTINE DO_RUN_SUM_SUB (idim, arg,
     .				   com_lo, com_hi,
     .				   com, com_mr, com_cx,
     .				   res, res_mr, res_cx )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* sum the component along axis idim producing a "running sum"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:  1/27/92 - based on DO_INT_INDEF_SUB
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xprog_state.cmn'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations:
	INTEGER	idim, com_lo, com_hi, com_mr, com_cx, res_mr, res_cx
	REAL	arg,
     .          com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* internal variable declarations:
	INTEGER i, j, k, l, m, n, grid, lo, hi, axis, ii, jj, 
     .          kk, ll, mm, nn
	REAL	bad_com, bad_res
	REAL*8	dsum

* initialize
	grid = mr_grid( com_mr )
	axis = grid_line( idim, grid )
	lo   = cx_lo_ss( com_cx, idim )
	hi   = cx_hi_ss( com_cx, idim )
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )

* SUM
* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 210 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 210 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 210 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 210 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 210 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	      dsum = 0.0D0
	      DO 205 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO 200 ii = i, hi
 200	            res(ii,j,k,l,m,n) = bad_res
	            GOTO 210
	         ENDIF
	         dsum = dsum + com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
 205	      CONTINUE
 210	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 310 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 310 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 310 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 310 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 310 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      dsum = 0.0D0
	      DO 305 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO 300 jj = j, hi
 300	            res(i,jj,k,l,m,n) = bad_res
	            GOTO 310
	         ENDIF
	         dsum = dsum + com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
 305	      CONTINUE
 310	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 410 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 410 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 410 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 410 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 410 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      dsum = 0.0D0
	      DO 405 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO 400 kk = k, hi
 400	            res(i,j,kk,l,m,n) = bad_res
	            GOTO 410
	         ENDIF
	         dsum = dsum + com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
 405	      CONTINUE
 410	   CONTINUE

* ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO 510 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 510 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 510 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 510 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 510 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      dsum = 0.0D0
	      DO 505 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO 500 ll = l, hi
 500	            res(i,j,k,ll,m,n) = bad_res
	            GOTO 510
	         ENDIF
	         dsum = dsum + com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
 505	      CONTINUE
 510	   CONTINUE

* ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO 610 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 610 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 610 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 610 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 610 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      dsum = 0.0D0
	      DO 605 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO 600 mm = m, hi
 600	            res(i,j,k,l,mm,n) = bad_res
	            GOTO 610
	         ENDIF
	         dsum = dsum + com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
 605	      CONTINUE
 610	   CONTINUE

* ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO 710 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 710 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 710 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 710 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 710 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      dsum = 0.0D0
	      DO 705 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_com ) THEN
* ... reject all data after a bad point
	            DO 700 nn = n, hi
 700	            res(i,j,k,l,m,nn) = bad_res
	            GOTO 710
	         ENDIF
	         dsum = dsum + com(i,j,k,l,m,n)
	         res(i,j,k,l,m,n) = dsum
 705	      CONTINUE
 710	   CONTINUE

	ENDIF

	RETURN
	END
