	INTEGER FUNCTION DO_4D_NGD(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					boxes	)

* call DO_4D_TRANS with the appropriate transformation

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V430: 7/96

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, com(*), res(*), boxes(*)

* internal variable declarations:
	INTEGER	CX_DIM_LEN

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* pass the transform to be performed
	CALL DO_4D_TRANS(   trans_4d_good_pt,
     .			    com, com_mr, com_cx,
     .			    res, res_mr, res_cx,
     .			    boxes )

	DO_4D_NGD = ferr_ok

	RETURN
	END	
