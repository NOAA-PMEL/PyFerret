	INTEGER FUNCTION DO_MIN( idim, arg,
     .				 com, com_mr, com_cx,
     .				 res, res_mr, res_cx,
     .				 buff	)

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
* find the minimum value of the component grid along axis idim

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/1/89
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include	'xprog_state.cmn'

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, buff(*),
     .          com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* internal variable declarations:
	INTEGER i, j, k, l, m, n, pt
	REAL	bad_res, minmum

* internal parameter declaration
	REAL		too_large
	PARAMETER     ( too_large = 1.E33 )

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing', isact_class_trans, res_cx, idim )

* initialize
	pt   = cx_lo_ss( res_cx, idim )
	bad_res = mr_bad_data( res_mr )

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 110 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 110 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 110 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 110 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	      minmum = too_large
	      DO 100 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) GOTO 100
	         IF ( com(i,j,k,l,m,n) .LT. minmum )
     .              minmum = com(i,j,k,l,m,n)
 100	      CONTINUE
	      IF ( minmum .EQ. too_large ) THEN
	         res(pt,j,k,l,m,n) = bad_res
	      ELSE
	         res(pt,j,k,l,m,n) = minmum
	      ENDIF
 110	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 210 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 210 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 210 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 210 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      minmum = too_large
	      DO 200 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) GOTO 200
	         IF ( com(i,j,k,l,m,n) .LT. minmum )
     .              minmum = com(i,j,k,l,m,n)
 200	      CONTINUE
	      IF ( minmum .EQ. too_large ) THEN
	         res(i,pt,k,l,m,n) = bad_res
	      ELSE
	         res(i,pt,k,l,m,n) = minmum
	      ENDIF
 210	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 310 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 310 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 310 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 310 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      minmum = too_large
	      DO 300 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) GOTO 300
	         IF ( com(i,j,k,l,m,n) .LT. minmum )
     .              minmum = com(i,j,k,l,m,n)
 300	      CONTINUE
	      IF ( minmum .EQ. too_large ) THEN
	         res(i,j,pt,l,m,n) = bad_res
	      ELSE
	         res(i,j,pt,l,m,n) = minmum
	      ENDIF
 310	   CONTINUE

* ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO 410 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 410 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 410 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 410 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 410 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      minmum = too_large
	      DO 400 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) GOTO 400
	         IF ( com(i,j,k,l,m,n) .LT. minmum )
     .              minmum = com(i,j,k,l,m,n)
 400	      CONTINUE
	      IF ( minmum .EQ. too_large ) THEN
	         res(i,j,k,pt,m,n) = bad_res
	      ELSE
	         res(i,j,k,pt,m,n) = minmum
	      ENDIF
 410	   CONTINUE

* ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO 510 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx)
	   DO 510 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 510 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 510 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 510 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      minmum = too_large
	      DO 500 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) GOTO 500
	         IF ( com(i,j,k,l,m,n) .LT. minmum )
     .              minmum = com(i,j,k,l,m,n)
 500	      CONTINUE
	      IF ( minmum .EQ. too_large ) THEN
	         res(i,j,k,l,pt,n) = bad_res
	      ELSE
	         res(i,j,k,l,pt,n) = minmum
	      ENDIF
 510	   CONTINUE

* ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO 610 m = cx_lo_s5(com_cx), cx_hi_s5(com_cx)
	   DO 610 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 610 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 610 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 610 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      minmum = too_large
	      DO 600 n = cx_lo_s6(com_cx), cx_hi_s6(com_cx) 
	         IF ( com(i,j,k,l,m,n) .EQ. bad_res ) GOTO 600
	         IF ( com(i,j,k,l,m,n) .LT. minmum )
     .              minmum = com(i,j,k,l,m,n)
 600	      CONTINUE
	      IF ( minmum .EQ. too_large ) THEN
	         res(i,j,k,l,m,pt) = bad_res
	      ELSE
	         res(i,j,k,l,m,pt) = minmum
	      ENDIF
 610	   CONTINUE

	ENDIF

	DO_MIN = ferr_ok

	RETURN
	END
