	SUBROUTINE XEQ_ELSE

* Process the ELSE in a multi-line IF compound statement

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420:  4/96

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* local variable declarations
	INTEGER	status

* initial sanity checks
	IF ( .NOT.if_conditional ) GOTO 5010

* check that something was given to "if" on
	IF ( num_args .GE. 1 ) GOTO 5100

* is this the end of the clause that has been executing?
	IF ( if_doing(ifstk) .EQ. pif_doing_clause ) THEN
	   if_doing(ifstk) = pif_skip_to_endif

* or the start of the clause to execute
	ELSEIF ( if_doing(ifstk) .EQ. pif_skip_to_clause ) THEN
	   if_doing(ifstk) = pif_doing_clause

* anything else is an internal error
	ELSE
	   GOTO 5200
	ENDIF

	RETURN
	
* error exit
* ... control stack pointer, CSP, corrected in GFD_ERROR_HANDLER
 5000	RETURN
 5010	CALL ERRMSG( ferr_invalid_command, status,
     .		'ELSE can only be used between IF and ENDIF', *5000)
 5100	CALL ERRMSG( ferr_syntax, status,
     .		'ELSE statement stands alone. Continue on next line"'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
 5200	CALL ERRMSG( ferr_internal, status, 'IF-ELSE', *5000)
	END
