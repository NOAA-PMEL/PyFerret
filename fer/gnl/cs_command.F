	SUBROUTINE CS_COMMAND( cmnd_line, * )

* this routine is called to get the next command when the program is under
* "internal" control (one command executing another)
* its function is to return the next command to execute or to flag that the
* internal level controlling the program has completed

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 4/29/87
* revision 0.1 - 4/12/88 - allowed negative delta for time axis
* v300: 1/2/93 - incorporated cmnd_semicolon
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V420:  4/25/96 - added IF command
*
* 1/98 sh&kob
* "KNOWN BUG: The proper formatting of the REPEAT limit requires a 
*              knowledge of the grid of the result(see lohi_string.F). 
*              Since the grid is not determined until the first repeat 
*              loop is executed the formatting of the first loop may be
*              incorrect. Fixing this would require that a new set of 
*              world coordinate formatting routines be developed that 
*              do not depend on the grid. This seemed too much trouble 
*              given the small consequeunces."



* calling argument declarations:
	CHARACTER*(*)	cmnd_line
* RETURN 1 - used when no command can be supplied

* internal variable declarations:
	INTEGER	prev, status, limit, i, nparen
        CHARACTER CX_DIM_STR*40

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/command.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xcontrol.cmn'
        include 'ferret_cmn/xrisc.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif

* internal error ?
	IF ( cs_cmnd_num( csp ) .EQ. cmnd_repeat ) THEN

***** REPEAT loop *******
* done yet ?
	   IF ( (cs_limit(csp)-cs_phase(csp))*cs_delta(csp)
     .          .LT. 0.0 ) THEN
* ... yes
	      CALL POP_CS
	      RETURN 1
	   ELSE
* ... no - get the next command
	      cmnd_line = cs_text( csp )
* ... set up the default context to evaluate the command
	      CALL CS_SET_CONTEXT
* ... prepare for next time around	   
	   cs_phase( csp ) = cs_phase( csp ) + cs_delta( csp )
	   ENDIF

* ... echo the progress so the operator can monitor
           IF ( mode_verify ) THEN
              risc_buff = CX_DIM_STR( cs_axis(csp), cx_last, ':',
     .             .TRUE., len_rbuff )
	      CALL SPLIT_LIST(pttmode_ops, ttout_lun, '!-> REPEAT: '
     .                           //risc_buff(1:len_rbuff), 12+len_rbuff)
           ENDIF

	ELSEIF ( cs_cmnd_num(csp) .EQ. cmnd_semicolon ) THEN

***** semicolon-separated command group
* initialize the extracting of a single command
           prev  = cs_phase(csp)
           limit = cs_limit(csp)

* done yet?
           IF ( prev .GE. limit ) THEN
* ... yes
	      CALL POP_CS
	      RETURN 1

	   ELSE
* ... no - get the next command (find semicolon NOT enclosed in parens)
              nparen = 0
              DO 100 i = prev+1, limit
                 IF ( cs_text(csp)(i:i) .EQ. '(' ) THEN
                    nparen = nparen + 1
                 ELSEIF ( cs_text(csp)(i:i) .EQ. ')' ) THEN
                    nparen = nparen - 1
                 ELSEIF ( cs_text(csp)(i:i) .EQ. ';' ) THEN
* ... found another semicolon-terminated command in group
                    IF ( nparen .EQ. 0 ) THEN
                       cmnd_line = cs_text(csp)(prev+1:i-1)
                       cs_phase(csp) = i
                       GOTO 1000
                    ENDIF
                 ENDIF
 100          CONTINUE
* ... last command in buffer
              cmnd_line = cs_text(csp)(prev+1:limit)
              cs_phase(csp) = limit
                   
	   ENDIF

* ****** conditional "IF" command

	ELSEIF ( cs_cmnd_num(csp) .EQ. cmnd_if ) THEN

* already done?
	   IF (cs_phase(csp) .GT. 0 ) THEN

* ... yes get outta here
	      CALL POP_CS
	      RETURN 1
	   ELSE

* ... no - get the next command
	      cmnd_line = cs_text( csp )
	      cs_phase(csp) = 1		! signal to POP after this command

	   ENDIF

* ***** unrecognized command in control of stack

        ELSE
           CALL ERRMSG( ferr_internal, status, 'command stack', *1000 )
        ENDIF

 1000   RETURN
	END
