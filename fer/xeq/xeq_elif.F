	SUBROUTINE XEQ_ELIF

* Process the intermediate logic of an ELIF ("ELSEIF") statement in a
* multi-line IF compound statement

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
	LOGICAL TRUE_OR_FALSE, condition
	INTEGER	STR_CASE_BLIND_COMPARE,
     .		vax_code, status

* initial sanity checks
	IF ( .NOT.if_conditional ) GOTO 5010

* is this the end of the clause that has been executing?
	IF ( if_doing(ifstk) .EQ. pif_doing_clause ) THEN
	   if_doing(ifstk) = pif_skip_to_endif
	   RETURN
	ENDIF

* more sanity checks
* (not the best error message ... but this is an internal never-happen error
* since pif_skip_to_endif is supposed to be trapped outside of this)
	IF ( if_doing(ifstk) .NE. pif_skip_to_clause ) GOTO 5010

* check that something was given to "if" on
	IF ( num_args .LT. 2 ) GOTO 5100

* optional "THEN" at end of line is OK
	IF ( num_args .GT. 1 ) THEN
	   IF ( num_args .GT. 2 ) GOTO 5300
	   vax_code = STR_CASE_BLIND_COMPARE(
     .				cmnd_buff(arg_start(2):arg_end(2)),
     .				'THEN' )
	   IF (vax_code .NE. vms_str_success) GOTO 5300
	ENDIF

* determine if the condition is true or false
	condition = TRUE_OR_FALSE(cmnd_buff(arg_start(1):arg_end(1)),
     .				  status)
	IF ( status .NE. ferr_ok ) GOTO 5000

* say which IF clause to execute
	IF ( condition ) THEN
	   if_doing(ifstk) = pif_doing_clause	! ELIF TRUE THEN (do it)
	ELSE
	   if_doing(ifstk) = pif_skip_to_clause	! ELIF FALSE THE (keep looking)
	ENDIF

	RETURN
	
* error exit
* ... control stack pointer, CSP, corrected in GFD_ERROR_HANDLER
 5000	RETURN
 5010	CALL ERRMSG( ferr_invalid_command, status,
     .		'ELIF can only be used between IF and ENDIF', *5000)
 5100	CALL ERRMSG( ferr_syntax, status, 'ELIF what ?', *5000)
 5300	CALL ERRMSG( ferr_syntax, status,
     .		'Entire ELIF statement should be "ELIF expr THEN"'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
	END
