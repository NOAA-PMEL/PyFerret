	INTEGER FUNCTION DO_MIN( idim, arg,
     .				 com, com_mr, com_cx,
     .				 res, res_mr, res_cx,
     .				 buff	)

* find the minimum value of the component grid along axis idim

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/1/89
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

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
	INTEGER i, j, k, l, pt
	REAL	bad_res, minmum

* internal parameter declaration
	REAL		too_large
	PARAMETER     ( too_large = 1.E33 )

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* initialize
	pt   = cx_lo_ss( res_cx, idim )
	bad_res = mr_bad_data( res_mr )

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 110 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 110 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	      minmum = too_large
	      DO 100 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 100
	         IF ( com(i,j,k,l) .LT. minmum ) minmum = com(i,j,k,l)
 100	      CONTINUE
	      IF ( minmum .EQ. too_large ) THEN
	         res(pt,j,k,l) = bad_res
	      ELSE
	         res(pt,j,k,l) = minmum
	      ENDIF
 110	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 210 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 210 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      minmum = too_large
	      DO 200 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 200
	         IF ( com(i,j,k,l) .LT. minmum ) minmum = com(i,j,k,l)
 200	      CONTINUE
	      IF ( minmum .EQ. too_large ) THEN
	         res(i,pt,k,l) = bad_res
	      ELSE
	         res(i,pt,k,l) = minmum
	      ENDIF
 210	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 310 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 310 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      minmum = too_large
	      DO 300 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 300
	         IF ( com(i,j,k,l) .LT. minmum ) minmum = com(i,j,k,l)
 300	      CONTINUE
	      IF ( minmum .EQ. too_large ) THEN
	         res(i,j,pt,l) = bad_res
	      ELSE
	         res(i,j,pt,l) = minmum
	      ENDIF
 310	   CONTINUE

* ALONG T AXIS
	ELSE
	   DO 410 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 410 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 410 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      minmum = too_large
	      DO 400 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 400
	         IF ( com(i,j,k,l) .LT. minmum ) minmum = com(i,j,k,l)
 400	      CONTINUE
	      IF ( minmum .EQ. too_large ) THEN
	         res(i,j,k,pt) = bad_res
	      ELSE
	         res(i,j,k,pt) = minmum
	      ENDIF
 410	   CONTINUE

	ENDIF

	DO_MIN = ferr_ok

	RETURN
	END	
