	INTEGER FUNCTION DO_SUM( idim, arg,
     .				 com, com_mr, com_cx,
     .				 res, res_mr, res_cx,
     .				 buff	)

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
* Boxes at the ends of the region will always be fully counted - no weighting
* based on the fraction of the box within the region as with integrals

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V230: 12/91 - based on DO_MAX
 
#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, buff(*),
     .		com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
        LOGICAL all_bad
	INTEGER i, j, k, l, pt
	REAL	bad_com, bad_res, sum

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* initialize
	pt   = cx_lo_ss( res_cx, idim )
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 110 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 110 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
              all_bad = .TRUE.
              sum = 0.0
	      DO 100 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 100
                 sum = sum + com(i,j,k,l)
                 all_bad = .FALSE.
 100	      CONTINUE
	      IF ( all_bad ) THEN
	         res(pt,j,k,l) = bad_res
	      ELSE
	         res(pt,j,k,l) = sum
	      ENDIF
 110	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 210 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 210 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              all_bad = .TRUE.
              sum = 0.0
	      DO 200 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 200
                 sum = sum + com(i,j,k,l)
                 all_bad = .FALSE.
 200	      CONTINUE
	      IF ( all_bad ) THEN
	         res(i,pt,k,l) = bad_res
	      ELSE
	         res(i,pt,k,l) = sum
	      ENDIF
 210	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 310 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 310 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              all_bad = .TRUE.
              sum = 0.0
	      DO 300 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 300
                 sum = sum + com(i,j,k,l)
                 all_bad = .FALSE.
 300	      CONTINUE
	      IF ( all_bad ) THEN
	         res(i,j,pt,l) = bad_res
	      ELSE
	         res(i,j,pt,l) = sum
	      ENDIF
 310	   CONTINUE

* ALONG T AXIS
	ELSE
	   DO 410 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 410 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 410 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              all_bad = .TRUE.
              sum = 0.0
	      DO 400 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) GOTO 400
                 sum = sum + com(i,j,k,l)
                 all_bad = .FALSE.
 400	      CONTINUE
	      IF ( all_bad ) THEN
	         res(i,j,k,pt) = bad_res
	      ELSE
	         res(i,j,k,pt) = sum
	      ENDIF
 410	   CONTINUE

	ENDIF

	DO_SUM = ferr_ok

	RETURN
	END	



