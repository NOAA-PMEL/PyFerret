	INTEGER FUNCTION DO_RUN_SUM(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .                                  box )

* sum the component along axis idim producing a "running sum"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:  1/27/92 - based on DO_INT_INDEF

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, box(*), com(*), res(*)

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* forward and backwards derivatives are identical except for a shift 
	CALL DO_RUN_SUM_SUB(	idim, arg,
     .				cx_lo_ss(com_cx,idim), cx_hi_ss(com_cx,idim),
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx )

* always success
	DO_RUN_SUM = ferr_ok

	RETURN
	END	
