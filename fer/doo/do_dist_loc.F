	INTEGER FUNCTION DO_DIST_LOC(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					buff	)

* locate the indicated argument value along the indicated axis and leave
* behind an "interpolating kernel": linear interpolating weights on the
* points immediately before and after the located value

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:  1/22/92 - new routine based on DO_FILL_AVE and DO_LOCATE

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
        LOGICAL first_pt, first_lt, crossing_found
	INTEGER i, j, k, l, ss1, lo_sub(4), hi_sub(4)
	REAL	bad_res, com1, com2, del

* internal (convenience) equivalences
	INTEGER	lo_s1, lo_s2, lo_s3, lo_s4, hi_s1, hi_s2, hi_s3, hi_s4
	EQUIVALENCE (lo_s1, lo_sub(1)) , (lo_s2, lo_sub(2)),
     .		    (lo_s3, lo_sub(3)) , (lo_s4, lo_sub(4)),
     .		    (hi_s1, hi_sub(1)) , (hi_s2, hi_sub(2)),
     .		    (hi_s3, hi_sub(3)) , (hi_s4, hi_sub(4))

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* initialize
	bad_res = mr_bad_data( com_mr )
        
* calculation limits
	DO 10 i = 1, 4
	   lo_sub(i) = cx_lo_ss(res_cx,i)
	   hi_sub(i) = cx_hi_ss(res_cx,i)
 10	CONTINUE

* SEARCH ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 l = lo_s4, hi_s4
	   DO 110 k = lo_s3, hi_s3
	   DO 110 j = lo_s2, hi_s2
              first_pt = .TRUE.
              crossing_found = .FALSE.
	      DO 100 i = lo_s1, hi_s1
                 res(i,j,k,l) = bad_res
                 IF ( crossing_found ) GOTO 100
                 IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 100
                 IF ( com(i,j,k,l).EQ.arg ) THEN
* ... exact equality
                    res(i,j,k,l) = 1.0
                    crossing_found = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l) .LT. arg
                    ss1 = i
                 ELSEIF ( first_lt .EQV. (com(i,j,k,l).LT.arg) ) THEN
* ... still have not crossed from GT to LT or visa versa
                    ss1 = i
                 ELSE
* ... found a "crossing"
                    com1 = com(ss1,j,k,l)
                    com2 = com(i  ,j,k,l)
                    del = com2 - com1
                    res(ss1,j,k,l) = (com2-arg) / del
                    res(i  ,j,k,l) = (arg-com1) / del
                    crossing_found = .TRUE.
                 ENDIF
 100          CONTINUE
 110	   CONTINUE

* SEARCH ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = lo_s4, hi_s4
	   DO 210 k = lo_s3, hi_s3
	   DO 210 i = lo_s1, hi_s1
              first_pt = .TRUE.
              crossing_found = .FALSE.
	      DO 200 j = lo_s2, hi_s2
                 res(i,j,k,l) = bad_res
                 IF ( crossing_found ) GOTO 200
                 IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 200
                 IF ( com(i,j,k,l).EQ.arg ) THEN
* ... exact equality
                    res(i,j,k,l) = 1.0
                    crossing_found = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l) .LT. arg
                    ss1 = j
                 ELSEIF ( first_lt .EQV. (com(i,j,k,l).LT.arg) ) THEN
* ... still have not crossed from GT to LT or visa versa
                    ss1 = j
                 ELSE
* ... found a "crossing"
                    com1 = com(i,ss1,k,l)
                    com2 = com(i,j  ,k,l)
                    del = com2 - com1
                    res(i,ss1,k,l) = (com2-arg) / del
                    res(i,j  ,k,l) = (arg-com1) / del
                    crossing_found = .TRUE.
                ENDIF
 200         CONTINUE
 210      CONTINUE

* SEARCH ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = lo_s4, hi_s4
	   DO 310 j = lo_s2, hi_s2
	   DO 310 i = lo_s1, hi_s1
              first_pt = .TRUE.
              crossing_found = .FALSE.
	      DO 300 k = lo_s3, hi_s3
                 res(i,j,k,l) = bad_res
                 IF ( crossing_found ) GOTO 300
                 IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 300
                 IF ( com(i,j,k,l).EQ.arg ) THEN
* ... exact equality
                    res(i,j,k,l) = 1.0
                    crossing_found = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l) .LT. arg
                    ss1 = k
                 ELSEIF ( first_lt .EQV. (com(i,j,k,l).LT.arg) ) THEN
* ... still have not crossed from GT to LT or visa versa
                    ss1 = k
                 ELSE
* ... found a "crossing"
                    com1 = com(i,j,ss1,l)
                    com2 = com(i,j,k  ,l)
                    del = com2 - com1
                    res(i,j,ss1,l) = (com2-arg) / del
                    res(i,j,k,  l) = (arg-com1) / del
                    crossing_found = .TRUE.
                ENDIF
 300         CONTINUE
 310      CONTINUE

* SEARCH ALONG T AXIS
	ELSE
	   DO 410 k = lo_s3, hi_s3
	   DO 410 j = lo_s2, hi_s2
	   DO 410 i = lo_s1, hi_s1
              first_pt = .TRUE.
              crossing_found = .FALSE.
	      DO 400 l = lo_s4, hi_s4
                 res(i,j,k,l) = bad_res
                 IF ( crossing_found ) GOTO 400
                 IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 400
                 IF ( com(i,j,k,l).EQ.arg ) THEN
* ... exact equality
                    res(i,j,k,l) = 1.0
                    crossing_found = .TRUE.
                 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = com(i,j,k,l) .LT. arg
                    ss1 = l
                 ELSEIF ( first_lt .EQV. (com(i,j,k,l).LT.arg) ) THEN
* ... still have not crossed from GT to LT or visa versa
                    ss1 = l
                 ELSE
* ... found a "crossing"
                    com1 = com(i,j,k,ss1)
                    com2 = com(i,j,k,l  )
                    del = com2 - com1
                    res(i,j,k,ss1) = (com2-arg) / del
                    res(i,j,k,l  ) = (arg-com1) / del
                    crossing_found = .TRUE.
                ENDIF
 400         CONTINUE
 410      CONTINUE
       ENDIF

* always success
	DO_DIST_LOC = ferr_ok
	RETURN

	END	
