	INTEGER FUNCTION DO_LOCATE( idim, arg,
     .				 com, com_mr, com_cx,
     .				 res, res_mr, res_cx,
     .				 buff	)

* find the world coordinate at which the component grid equals the argument

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  1/11/90
* V230:  1/22/92 - fixed bug of missing exact equality

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
	LOGICAL	first_pt, first_lt
	INTEGER i, j, k, l, pt, ss1, grid
	REAL	bad_res, ww1, ww2, del, com1, com2
	REAL*8	TM_WORLD

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* initialize
	grid = cx_grid( com_cx )
	pt   = cx_lo_ss( res_cx, idim )
	bad_res = mr_bad_data( res_mr )

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 110 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 110 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	      first_pt = .TRUE.
	      DO 100 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 100
                 IF ( com(i,j,k,l) .EQ. arg ) THEN
                    res(pt,j,k,l) = TM_WORLD(i, grid, x_dim, box_middle)
	            GOTO 110
	         ELSEIF ( first_pt ) THEN
	           first_pt = .FALSE.
		   first_lt = com(i,j,k,l) .LT. arg
	           ss1 = i
		 ELSEIF ( first_lt .EQV. (com(i,j,k,l).LT.arg) ) THEN
	           ss1 = i
		 ELSE
* ... found a "zero-crossing"
	            ww1 = TM_WORLD ( ss1, grid, x_dim, box_middle )
	            ww2 = TM_WORLD ( i,   grid, x_dim, box_middle )
	            com1 = com(ss1,j,k,l)
	            com2 = com(i  ,j,k,l)
	            del = com2 - com1
	            res(pt,j,k,l) = ( ww2*(arg-com1) + ww1*(com2-arg) ) / del
	            GOTO 110
	         ENDIF
 100	      CONTINUE
* ... no zero-crossing found
	      res(pt,j,k,l) = bad_res
 110	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 210 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 210 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      first_pt = .TRUE.
	      DO 200 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 200
                 IF ( com(i,j,k,l) .EQ. arg) THEN
	            res(i,pt,k,l) = TM_WORLD( j, grid, y_dim, box_middle )
	            GOTO 210
	         ELSEIF ( first_pt ) THEN
	           first_pt = .FALSE.
		   first_lt = com(i,j,k,l) .LT. arg
	           ss1 = j
		 ELSEIF ( first_lt .EQV. (com(i,j,k,l).LT.arg) ) THEN
	           ss1 = j
		 ELSE
* ... found a "zero-crossing"
	            ww1 = TM_WORLD ( ss1, grid, y_dim, box_middle )
	            ww2 = TM_WORLD ( j,   grid, y_dim, box_middle )
	            com1 = com(i,ss1,k,l)
	            com2 = com(i,j  ,k,l)
	            del = com2 - com1
	            res(i,pt,k,l) = ( ww2*(arg-com1) + ww1*(com2-arg) ) / del
	            GOTO 210
	         ENDIF
 200	      CONTINUE
* ... no zero-crossing found
	      res(i,pt,k,l) = bad_res
 210	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 310 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 310 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      first_pt = .TRUE.
	      DO 300 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 300
                 IF ( com(i,j,k,l) .EQ. arg ) THEN
	            res(i,j,pt,l) = TM_WORLD ( k, grid, z_dim, box_middle )
	            GOTO 310
	         ELSEIF ( first_pt ) THEN
	           first_pt = .FALSE.
		   first_lt = com(i,j,k,l) .LT. arg
	           ss1 = k
		 ELSEIF ( first_lt .EQV. (com(i,j,k,l).LT.arg) ) THEN
	           ss1 = k
		 ELSE
* ... found a "zero-crossing"
	            ww1 = TM_WORLD ( ss1, grid, z_dim, box_middle )
	            ww2 = TM_WORLD ( k,   grid, z_dim, box_middle )
	            com1 = com(i,j,ss1,l)
	            com2 = com(i,j,k  ,l)
	            del = com2 - com1
	            res(i,j,pt,l) = ( ww2*(arg-com1) + ww1*(com2-arg) ) / del
	            GOTO 310
	         ENDIF
 300	      CONTINUE
* ... no zero-crossing found
	      res(i,j,pt,l) = bad_res
 310	   CONTINUE

* ALONG T AXIS
	ELSE
	   DO 410 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 410 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 410 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	      first_pt = .TRUE.
	      DO 400 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_res ) GOTO 400
                 IF ( com(i,j,k,l) .EQ. arg ) THEN
	            res(i,j,k,pt) = TM_WORLD( l, grid, t_dim, box_middle )
	            GOTO 410
	         ELSEIF ( first_pt ) THEN
	           first_pt = .FALSE.
		   first_lt = com(i,j,k,l) .LT. arg
	           ss1 = l
		 ELSEIF ( first_lt .EQV. (com(i,j,k,l).LT.arg) ) THEN
	           ss1 = l
		 ELSE
* ... found a "zero-crossing"
	            ww1 = TM_WORLD ( ss1, grid, t_dim, box_middle )
	            ww2 = TM_WORLD ( l,   grid, t_dim, box_middle )
	            com1 = com(i,j,k,ss1)
	            com2 = com(i,j,k,l  )
	            del = com2 - com1
	            res(i,j,k,pt) = ( ww2*(arg-com1) + ww1*(com2-arg) ) / del
	            GOTO 410
	         ENDIF
 400	      CONTINUE
* ... no zero-crossing found
	      res(i,j,k,pt) = bad_res
 410	   CONTINUE

	ENDIF

	DO_LOCATE = ferr_ok

	RETURN
	END	
