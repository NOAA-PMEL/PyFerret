	SUBROUTINE IS_RECONCILE( memory, *, *, status )

* reconcile any conflicts between the component requested and the one received
* for example, LIST/I=3:7  test[i=2:5]

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420: 11/95

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	INTEGER	status
	REAL	memory( mem_blk_size, max_mem_blks )
* normal return is for error conditions
* exit #1 is to request another grid ( stack has been pushed )
* exit #2 is to signal completion of the activity at this stack level

*   ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level of last activity ( component just computed )
*	 isp	<--	current stack level (for moduloing )

* internal variable declarations:
	LOGICAL	by_ss(4), given(4)
	INTEGER i, mr, com_cx, res_cx,	com_isp, res_isp,
     .		com_lo, com_hi, res_lo(4), res_hi(4)
	CHARACTER VAR_CODE*24, qual*1

* --- end of introductory code ---

* initialize
	res_isp    = isp		! result (what the user requested)
	com_isp    = isp + 1		! component (what we have received)
	res_cx     = is_cx( res_isp )
	com_cx     = is_cx( com_isp )
	mr	   = is_mr( com_isp )

* defaults
	is_mr( res_isp ) = mr
	is_cx( res_isp ) = res_cx

* save the key ingredients from the requesting context (res_cx)
* (FLESH_OUT_AXIS has already been applied so world coords need not be used)
	DO 10 i = 1, 4
	   res_lo(i) =  cx_lo_ss(res_cx,i)
	   res_hi(i) =  cx_hi_ss(res_cx,i)
	   by_ss(i)  =  cx_by_ss(i,res_cx)
	   given(i)  =  cx_given(i,res_cx)
 10	CONTINUE

* transfer the full context (which includes data set, etc.)
	CALL TRANSFER_CONTEXT( com_cx, res_cx )

* check each axis for possible reconciliation
	DO 100 i = 1, 4
	   IF ( given(i) ) THEN
	      com_lo = cx_lo_ss(com_cx,i)
	      com_hi = cx_hi_ss(com_cx,i)
	      IF ( com_lo    .EQ. unspecified_int4
     .	      .OR. res_lo(i) .EQ. unspecified_int4 ) GOTO 100
	      IF (res_lo(i).NE.com_lo .OR. res_hi(i).NE.com_hi) THEN

* unreconcilable limits
	         IF (res_lo(i).GT.com_hi .OR. res_hi(i).LT.com_lo) THEN
	            IF ( by_ss(i) ) THEN
	               qual = ss_dim_name(i) 
	            ELSE
	               qual = ww_dim_name(i) 
	            ENDIF
	            CALL WARN('Invalid '//qual//
     .			' axis limits ignored: '//
     .			 VAR_CODE(cx_category(res_cx),cx_variable(res_cx)) )

	         ELSE
* diagnostic mode output: " doing reconcile VAR_NAME[x=lo:hi,D=#]"
	            IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_reconcile, com_cx, i )

* trim to limits that satisfy both component and result
	            cx_lo_ss(res_cx,i) = MAX(com_lo, res_lo(i))
	            cx_hi_ss(res_cx,i) = MIN(com_hi, res_hi(i))
	            cx_by_ss(i, res_cx) = .TRUE.
	            CALL FLESH_OUT_AXIS(i, res_cx, status )
	            IF ( status .NE. ferr_ok ) GOTO 5000
	         ENDIF
	      ENDIF
	   ENDIF
 100	CONTINUE

* success -- clean up
	cx_stack_ptr = cx_stack_ptr - 1 ! pop com_cx  from context stack
	CALL MR_NOT_IN_USE( mr )	! remove memory protection on component
	RETURN 2

* error exits
 5000	CALL MR_NOT_IN_USE( mr )	! remove memory protection on component
	RETURN
	END	
