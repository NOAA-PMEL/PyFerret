	INTEGER FUNCTION DO_NEAREST_DIST_BELOW(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					dummy )

* @CDB: call DO_NEAREST appropriate for closest distance BELOW

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 9/97

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	com(*), res(*), arg, dummy(*)

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

* request the actual transform
* ... CALL DO_NEAREST( idim, .FALSE.<==by_index , .FALSE.<==find_above

	CALL DO_NEAREST( idim, .FALSE., .FALSE.,
     .			 com, com_mr, com_cx,
     .			 res, res_mr, res_cx )

	DO_NEAREST_DIST_BELOW = ferr_ok

	RETURN
	END	
