	INTEGER FUNCTION DO_DERIV_FRWD(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					box	)

* compute forward derivative of component along axis idim

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/6/89
*	10/16/89 - changes to adjustable array declarations
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, box(*), com(*), res(*)

* internal variable declarations
	INTEGER	ishf		!kob 12/96
	PARAMETER (ishf = 0)	! 0 for forward, 1 for backward
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
	CALL DD1STEP (  idim, ishf,
     .			cx_lo_ss(com_cx,idim),cx_hi_ss(com_cx,idim), box,
     .			com, com_mr,
     .			res, res_mr, res_cx )

* always success
	DO_DERIV_FRWD = ferr_ok

	RETURN
	END	
