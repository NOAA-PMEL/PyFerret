	SUBROUTINE CS_SEMICOLON( cmnd_line, * )

* this routine is called to save a group of commands separated by semicolons
* and bump the FERRET "control stack" to place that group of commands in
* control

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V3.00 - 2/1/93

* calling argument declarations:
	CHARACTER*(*)	cmnd_line
* RETURN 1 - always taken

* internal variable declarations:
	INTEGER	TM_LENSTR1, status

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/command.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xcontrol.cmn'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
#endif

* allocate next spot on the control stack
* ... but leave the stack in place in case of errors in this routine
	CALL STACK_PTR_UP( csp, cs_max, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* load the command group
	cs_text (csp) = cmnd_line
        cs_phase(csp) = 0    ! points before the first character
        cs_limit(csp) = TM_LENSTR1( cmnd_line )

* and tell 'em who owns this level
	cs_cmnd_num( csp ) = cmnd_semicolon

* tell 'em we've taken over and start workin'
	cs_in_control = .TRUE.
	input_source  = input_from_cs
	RETURN 1

* error exit
* ... control stack pointer, CSP, corrected in ERRMSG
 5000	RETURN
	END
