	INTEGER FUNCTION DO_SMTH_WELCH(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					wt	)

* smooth along axis idim by a Welch filter (window)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* for Unix/RISC port 9/91 - based on DO_SMTH_BINML

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, wt(*), com(*), res(*)

* internal variable declarations
	LOGICAL	even
	INTEGER	i, wlen, status
	REAL	weight

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

* initialize
	wlen = arg	! length of smoother
!       IF ( wlen .GE. iswksiz ) CALL ERRMSG(ferr_prog_limit, status,
!     .                 'smoother too long', *5000 )
        IF ( MOD(wlen,2).EQ.0 )  CALL ERRMSG(ferr_out_of_range, status,
     .                  'Welch smoother length must be odd', *5000 )

* create weights array to be convolved with data
	CALL WELCH_WT( wt, wlen )

* now convolve this along the desired axis of the grid
	CALL CONVOLVE(	idim, wlen/2, wt,
     .			com, com_mr, com_cx,
     .			res, res_mr, res_cx )

* success
	DO_SMTH_WELCH = ferr_ok
	RETURN

* error
 5000	DO_SMTH_WELCH = status
	RETURN
	END	
