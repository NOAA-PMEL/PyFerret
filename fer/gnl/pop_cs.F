	SUBROUTINE POP_CS

* POP the contral stack turning control over to the next lower level, if any

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 4/29/87
* V230: restore original context following REPEAT command
* V300 - 2/2/93 - added semicolon-separated command groups
* V420 - added IF (single line version controls input source)
*      - generate warning if multi-line IF is unclosed

* internal variable declarations:
	INTEGER	LIB_FREE_LUN, stat, axis

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/command.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xcontrol.cmn'
        include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
        INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* internal error ?
	IF ( csp .LT. 1 ) STOP 'csp_undflw'

* popping out of a stack level with an unclosed IF statement is a form of error
* we will issue a warning and then implicitly close the IF
	IF (if_conditional) THEN
	   IF (csp .LE. if_cs(ifstk) ) THEN
	      CALL WARN('Unclosed IF statement ... assuming ENDIF')
	      ifstk = ifstk - 1
	      IF (ifstk .EQ. 0 ) if_conditional = .FALSE.
	   ENDIF
	ENDIF

* clean up old level
	IF ( cs_cmnd_num( csp ) .EQ. cmnd_go ) THEN
* ... close input GO file
	   CLOSE( UNIT = cmnd_lun, ERR = 10 )
 10	   stat = LIB_FREE_LUN( cmnd_lun )
	ELSEIF ( cs_cmnd_num( csp ) .EQ. cmnd_repeat ) THEN
* ... restore context prior to REPEAT loop (along repeat axis only)
           axis = cs_axis( csp )
           cx_by_ss(axis,cx_last) = cs_save_by_ss( csp )
           IF ( cs_save_by_ss( csp ) ) THEN
              cx_lo_ss (cx_last,axis) = cs_save_lo( csp )
              cx_hi_ss (cx_last,axis) = cs_save_hi( csp )
           ELSE
              cx_lo_ww (axis,cx_last) = cs_save_lo( csp )
              cx_hi_ww (axis,cx_last) = cs_save_hi( csp )
           ENDIF
	ENDIF
	csp = csp - 1

* now who gets control ?
	IF ( csp .EQ. 0 ) THEN
* ... the operator
	   cs_in_control = .FALSE.
	   input_source  = input_from_tty
	ELSE
* ... the next stack level down
	   IF    ( cs_cmnd_num( csp ) .EQ. cmnd_go     ) THEN
	      input_source = input_from_file
	      cmnd_lun = cs_lun( csp )
	   ELSEIF ( cs_cmnd_num( csp ) .EQ. cmnd_semicolon
     .         .OR. cs_cmnd_num( csp ) .EQ. cmnd_if
     .         .OR. cs_cmnd_num( csp ) .EQ. cmnd_repeat ) THEN
	      input_source = input_from_cs
	   ELSE
	      STOP 'cs_stk_??'
	   ENDIF
	ENDIF
		   
	RETURN
	END




