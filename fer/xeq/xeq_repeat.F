	SUBROUTINE XEQ_REPEAT

* sets up the control stack (COMMON/~CONTROL/) to repeat the indicated
* command for a range of default contexts
* eg. "REPEAT/T=27740:35000:10 CONTOUR/@N/K=1 TEMP"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/29/87
* revision 0.1 - 9/16/87 - unknown_qual_bad moved to FERRET.PARM
* revision 0.2 - 9/25/87 - added "d" notation for REPEAT limits
* revision 0.3 - 4/11/88 - special checks on T axis for calendar notation
* revision 0.4 - 7/27/88 - added default context to GET_CONTEXT_MODS call
* revision 0.5 -10/31/88 - limits reversed for negative delta
* V200:  6/15/89 - eliminated "d" notation and "relative" logic
* Unix/RISC port - 1/91 - cant use "/star" in documentation
* V230:  7/23/92 - added cs_save_* to restore context after REPEAT loops

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/command.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xcontrol.cmn'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* local variable declarations
	LOGICAL	by_date
	INTEGER	i, idim, limits_cx, status, csp1
	REAL*8	temp

* check that something was given to repeat
	IF ( num_args .EQ. 0 ) CALL ERRMSG
     .			( ferr_invalid_command, status, 'REPEAT what ?', *5000 )

* get the limits for repeating
*  eg. "REPEAT/X=160E:140W:10 LIST/X=-5D:+5D/Y=0D:5D TEMP"
	limits_cx = cx_stack_ptr_base + 1
	CALL INIT_CONTEXT_MODS ( limits_cx )
	CALL GET_CONTEXT_MODS ( cx_last,
     .				cmnd_buff,
     .				limits_cx,
     .				max_qual_list,
     .				num_qualifiers,
     .				qual_start,
     .				qual_end,
     .				unknown_qual_bad,
     .				status	)
	IF ( status .NE. ferr_ok ) GOTO 5000

* determine the (unique) axis of REPEATing
	idim = unspecified_int4
	DO 100 i = 1,4
	   IF ( cx_given(i,limits_cx) ) THEN
	      IF ( idim .EQ. unspecified_int4 ) THEN
	         idim = i
	      ELSE
	         CALL ERRMSG( ferr_invalid_command, status,
     .			'Use nested REPEAT for multiple axes', *5000 )
	      ENDIF
 	   ENDIF
 100	CONTINUE
	IF ( idim .EQ. unspecified_int4 ) CALL ERRMSG
     .		(ferr_invalid_command,status,'REPEAT on which axis ?',*5000)

* allocate next spot for the given limits on the control stack
* ... but leave the stack in place in case of errors in this routine
	CALL STACK_PTR_UP( csp, cs_max, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	csp1 = csp
	CALL STACK_PTR_DN( csp, 0, status )

* load the start and end limits and the increment
	cs_by_ss( csp1 ) = cx_by_ss( idim, limits_cx )
	IF ( cx_delta( idim, limits_cx ) .EQ. unspecified_val8 ) THEN
	   cs_delta( csp1 ) = 1
	ELSE
	   cs_delta( csp1 ) = cx_delta( idim, limits_cx )
	ENDIF
	IF ( cs_by_ss( csp1 ) ) THEN
	   cs_limit( csp1 ) = cx_hi_ss( limits_cx, idim )
	   cs_phase( csp1 ) = cx_lo_ss( limits_cx, idim )
	ELSE
	   cs_limit( csp1 ) = cx_hi_ww( idim, limits_cx )
	   cs_phase( csp1 ) = cx_lo_ww( idim, limits_cx )
	ENDIF

* save the current context so it can be restored afterwards
        cs_save_by_ss( csp1 ) = cx_by_ss(idim,cx_last)
        IF ( cs_save_by_ss(csp1) ) THEN
           cs_save_lo( csp1 ) = cx_lo_ss (cx_last,idim)
           cs_save_hi( csp1 ) = cx_hi_ss (cx_last,idim)
        ELSE
           cs_save_lo( csp1 ) = cx_lo_ww (idim,cx_last)
           cs_save_hi( csp1 ) = cx_hi_ww (idim,cx_last)
        ENDIF

* special check for the validity of time axis limits: calendar dates are
* stored as negative seconds since 1-jan-0000:00:00:00
	by_date = ( idim .EQ. t_dim )
     .	.AND. ( .NOT.cx_by_ss(idim,limits_cx) )
     .	.AND. ( cx_lo_ww(idim,limits_cx) .LT. 0.0D0 )
* ... delta value in hours: hours-->seconds (to add to secs from BC)
	IF ( by_date ) cs_delta(csp1) =  -3600. * cs_delta(csp1)

* reverse limits if negative delta is desired
	IF ( ( .NOT.by_date .AND. (cs_delta(csp1) .LT. 0.0D0) )
     .	.OR. (      by_date .AND. (cs_delta(csp1) .GT. 0.0D0) ) ) THEN
	   temp = cs_phase(csp1)
	   cs_phase(csp1) = cs_limit(csp1)
	   cs_limit(csp1) = temp
	ENDIF

* from here on no errors can occur so make the stack bump permanent
	CALL STACK_PTR_UP( csp, cs_max, status )

* load the axis to which the limits apply
	cs_axis( csp ) = idim

* load the command line to be repeatedly executed
	cs_text( csp ) = cmnd_buff( arg_start(1):len_cmnd )

* and tell 'em who owns this level
	cs_cmnd_num( csp ) = cmnd_repeat

* and set the default context in preparation for first call
	CALL CS_SET_CONTEXT

* success - tell 'em we've taken over and start workin'
	cs_in_control = .TRUE.
	input_source  = input_from_cs
	RETURN

* error exit
* ... control stack pointer, CSP, corrected in GFD_ERROR_HANDLER
 5000	RETURN
	END
