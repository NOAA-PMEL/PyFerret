	INTEGER FUNCTION DO_AVERAGE(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					box	)

* average the component grid into the result grid
* (done by calling a single routine used for both averaging and integrating)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/1/89

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, com(*), res(*), box(*)

* internal variable declarations:
* ...

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

	CALL DO_AVE_INT( isact_average, idim,
     .			 cx_lo_ss(com_cx,idim), cx_hi_ss(com_cx,idim),
     .			 com, com_mr, com_cx,
     .			 res, res_mr, res_cx,
     .			 box )

	DO_AVERAGE = ferr_ok

	RETURN
	END	
