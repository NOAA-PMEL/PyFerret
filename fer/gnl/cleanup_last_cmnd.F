	SUBROUTINE CLEANUP_LAST_CMND( control_stack_level )

* clean up the mess left by the previous command - esp. preparing a buffer
* of messages for the calling program, if any

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V312:  5/19/94 - with code extracted from GET_COMMAND
* V320:  11/94   - dont process error message buffer -- done in ferret_dispatch

* calling argument declarations: (none)
	INTEGER	control_stack_level

* internal variable declarations:
	INTEGER status

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/command.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
	include 'ferret_cmn/xcontrol.cmn'
	include 'ferret_cmn/xinterrupt.cmn'
	include 'ferret_cmn/xgui.cmn'
#else
	INTEGER	SMG$READ_COMPOSED_LINE, vax_code

	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'
	INCLUDE 'FERRET_CMN:XGUI.CMN'
#endif

* re-store ^C interrupt handler following interrupt
 10	IF ( interrupted ) CALL ERRMSG
     .				( ferr_interrupt, status, ' ', *10 )
	ppl_interrupted = .FALSE.	! for interrupt of PPLUS

	IF ( mode_xwindows ) CALL PROCESS_X_EVENTS    ! 3/94

* send the state of the control stack back to the caller
* (zero is command line level)
	control_stack_level = csp

* this needed for logic in ferret_dispatch
	cmnd_buff = ' '

	RETURN
	END
