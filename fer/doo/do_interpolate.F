	INTEGER FUNCTION DO_INTERPOLATE( axis, arg,
     .					 com, com_mr, com_cx,
     .					 res, res_mr, res_cx,
     .					 work	)

* interpolate the component grid along the given axis
* note: if requested interpolation lies beyond the first or last grid point
*	(but within the edge grid box) then the component context will be
*	only that single edge point (ss_lo=ss_hi).  The code below will
*	extrapolate the end value to the box edge.

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
	INTEGER	axis, com_mr, com_cx, res_mr, res_cx
	REAL	arg, work(*),
     .		com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	INTEGER i, j, k, l, ss, ss_lo, ss_hi
	REAL	bad, delta, c_lo, c_hi, z_lo, z_hi
	REAL*8	TM_WORLD, ww, ww_lo, ww_hi

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, axis )

* flags for bad/missing data
        bad = mr_bad_data(res_mr)

* bits and pieces for interpolation
	ss    = cx_lo_ss(res_cx,axis)
	ww    = cx_lo_ww(axis,res_cx)
	ss_lo = cx_lo_ss(com_cx,axis)
	ss_hi = cx_hi_ss(com_cx,axis)
	ww_lo = TM_WORLD( ss_lo, cx_grid(com_cx), axis, box_middle )
	ww_hi = TM_WORLD( ss_hi, cx_grid(com_cx), axis, box_middle )
	delta = ww_hi - ww_lo
	IF ( delta .EQ. 0.0 ) THEN
* ... extrapolate last grid point towards end box edge
	   c_lo = 1.0
	   c_hi = 0.0
	ELSE
* ... normal interpolation
	   c_lo    = REAL(ww_hi - ww) / delta
	   c_hi    = 1.0 - c_lo
	ENDIF

*       I N T E R P O L A T E
* ( separate loops for each axis )
	IF ( axis .EQ. x_dim ) THEN

	   DO 100 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 100 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 100 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)

* source data points
	      z_lo = com(ss_lo,j,k,l)
	      z_hi = com(ss_hi,j,k,l)

* interpolate
	      IF (z_lo .EQ. bad .OR. z_hi .EQ. bad ) THEN
	         res(ss,j,k,l) = bad
	      ELSE
	         res(ss,j,k,l) = c_lo*z_lo + c_hi*z_hi
	      ENDIF

 100	   CONTINUE

	ELSEIF ( axis .EQ. y_dim ) THEN

	   DO 200 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 200 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 200 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)

* source data points
	      z_lo = com(i,ss_lo,k,l)
	      z_hi = com(i,ss_hi,k,l)

* interpolate
	      IF (z_lo .EQ. bad .OR. z_hi .EQ. bad ) THEN
	         res(i,ss,k,l) = bad
	      ELSE
	         res(i,ss,k,l) = c_lo*z_lo + c_hi*z_hi
	      ENDIF

 200	   CONTINUE

	ELSEIF ( axis .EQ. z_dim ) THEN

	   DO 300 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 300 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 300 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)

* source data points
	      z_lo = com(i,j,ss_lo,l)
	      z_hi = com(i,j,ss_hi,l)

* interpolate
	      IF (z_lo .EQ. bad .OR. z_hi .EQ. bad ) THEN
	         res(i,j,ss,l) = bad
	      ELSE
	         res(i,j,ss,l) = c_lo*z_lo + c_hi*z_hi
	      ENDIF

 300	   CONTINUE

	ELSEIF ( axis .EQ. t_dim ) THEN

	   DO 400 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 400 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 400 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)

* source data points
	      z_lo = com(i,j,k,ss_lo)
	      z_hi = com(i,j,k,ss_hi)

* interpolate
	      IF (z_lo .EQ. bad .OR. z_hi .EQ. bad ) THEN
	         res(i,j,k,ss) = bad
	      ELSE
	         res(i,j,k,ss) = c_lo*z_lo + c_hi*z_hi
	      ENDIF

 400	   CONTINUE
	ENDIF

	DO_INTERPOLATE = ferr_ok

	RETURN
	END	
