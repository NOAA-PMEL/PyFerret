	SUBROUTINE DO_STRING_GOODBAD (	idim, trans,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx )

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
* Count the number of "FERRET Ver. 6.14" and "bad" values in the string array data,
* where FERRET Ver. 6.14 means length > 0


* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program


	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xprog_state.cmn'

* calling argument declarations:
	INTEGER	idim, trans, com_mr, com_cx, res_mr, res_cx
	REAL*8  com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit )
	REAL	res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	LOGICAL	good_un
	INTEGER GET_C_STRING_LEN, 
     .          ii, i, j, k, l, pt, ngd, nbd, slen
	REAL	bad_res

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* initialize
        pt   = cx_lo_ss( res_cx, idim )
        bad_res = mr_bad_data( res_mr )
	good_un  = trans .EQ. trans_good_pt

* Write a note about missing values of strings 

        IF (good_un) THEN
	   CALL WARN
     .    ('@NGD of string variable, counting null strings as missing')
        ELSE
	   CALL WARN
     .    ('@NBD of string variable, counting null strings as missing')
        ENDIF

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 110 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 110 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
              ngd = 0
              nbd = 0
	      DO 100 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         slen = GET_C_STRING_LEN(com(i,j,k,l))
		 IF ( slen .GT. 0 ) THEN
                    ngd = ngd + 1
                 ELSE
                    nbd = nbd + 1
                 ENDIF
 100	      CONTINUE
              IF ( good_un ) THEN
                 res(pt,j,k,l) = ngd
              ELSE
                 res(pt,j,k,l) = nbd
              ENDIF
 110	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 210 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 210 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              ngd = 0
              nbd = 0
	      DO 200 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         slen = GET_C_STRING_LEN(com(i,j,k,l))
		 IF ( slen .GT. 0 ) THEN
                    ngd = ngd + 1
                 ELSE
                    nbd = nbd + 1
                 ENDIF
 200	      CONTINUE
              IF ( good_un ) THEN
                 res(i,pt,k,l) = ngd
              ELSE
                 res(i,pt,k,l) = nbd
              ENDIF
 210	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 310 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 310 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              ngd = 0
              nbd = 0
	      DO 300 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         slen = GET_C_STRING_LEN(com(i,j,k,l))
		 IF ( slen .GT. 0 ) THEN
                    ngd = ngd + 1
                 ELSE
                    nbd = nbd + 1
                 ENDIF
 300	      CONTINUE
              IF ( good_un ) THEN
                 res(i,j,pt,l) = ngd
              ELSE
                 res(i,j,pt,l) = nbd
              ENDIF
 310	   CONTINUE

* ALONG T AXIS
	ELSE
	   DO 410 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 410 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 410 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              ngd = 0
              nbd = 0
	      DO 400 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         slen = GET_C_STRING_LEN(com(i,j,k,l))
		 IF ( slen .GT. 0 ) THEN
                    ngd = ngd + 1
                 ELSE
                    nbd = nbd + 1
                 ENDIF
 400	      CONTINUE
              IF ( good_un ) THEN
                 res(i,j,k,pt) = ngd
              ELSE
                 res(i,j,k,pt) = nbd
              ENDIF
 410	   CONTINUE
	ENDIF

* success
	RETURN

	END
