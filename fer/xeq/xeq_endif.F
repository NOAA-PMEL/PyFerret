	SUBROUTINE XEQ_ENDIF

* Process the ENDIF statement in a multi-line IF compound statement

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420:  4/96

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/command.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/gfdl_vms.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xcontrol.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* local variable declarations
	INTEGER	status

* initial sanity checks
	IF ( .NOT.if_conditional ) GOTO 5100

* check that something was given to "if" on
	IF ( num_args .GT. 0 ) GOTO 5300

* pop the IF stack
	IF (ifstk .LT. 1 ) GOTO 5400
	ifstk = ifstk - 1
	IF ( ifstk .EQ. 0 ) if_conditional = .FALSE.

	RETURN
	
* error exit
* ... control stack pointer, CSP, corrected in GFD_ERROR_HANDLER
 5000	RETURN
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		'ENDIF can only be used in an IF clause', *5000)
 5300	CALL ERRMSG( ferr_syntax, status,
     .		'Trash on ENDIF statement"'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
 5400	CALL ERRMSG( ferr_internal, status, 'ENDIF ifstk??', *5000)
	END
