	SUBROUTINE XEQ_PPLUS

* access program PPLUS ( the underlying graphics package ) directly - either
* by sending a single command to it or by entering the PPLUS command interpreter

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/7/87
* revision 0.1 - 9/8/88 - added /RESET option
* revision 0.2 - 1/31/89 - added PPLUS interruptibility
* Unix/RISC port - 10/9/91 - check to see if user changes ylen or yorg

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
	include 'ferret_cmn/xplot_setup.cmn'    ! added 10/9/91
        include 'tmap_pplv11inc/axis_inc.decl'  ! added 10/9/91 for ylen,yorg
        include 'pplv11inc/AXIS.INC'            ! added 10/9/91 for ylen,yorg
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN/XPLOT_SETUP.CMN'    ! added 10/9/91
        INCLUDE 'PPLV11INC:AXIS.INC'            ! added 10/9/91 for ylen,yorg
        INCLUDE 'PPLV11INC:AXIS.INC'            ! added 10/9/91 for ylen,yorg
#endif
	
* local variable declarations:
        REAL temp_ylen, temp_yorg

* local parameter declarations:
	INTEGER		slash_reset
	PARAMETER     ( slash_reset = 0+1 )

* turn on PLOT+ ?
	CALL START_PPLUS

* save ylen and yorg in case the user manually changes those values in PLOT+
* 10/9/91 addition 
        temp_ylen = ylen
        temp_yorg = yorg

	IF ( qual_given( slash_reset ) .GT. 0 ) THEN
* ... reset plot environment to defaults
	   CALL DISP_RESET
	ELSEIF ( num_args .GT. 0 ) THEN
* ... pass a single command to PLOT5
	   CALL PPLCMD (' ',' ',0,cmnd_buff(arg_start(1):len_cmnd),1,1)
	ELSE
* ... enter PPLUS interactively
#ifdef unix
	   CALL PPLCMD ( '/dev/tty', ' ', 0, ' ', 1, 0 )
#else
	   CALL PPLCMD ( 'TT:', ' ', 0, ' ', 1, 0 )
#endif
	   ppl_interrupted = .FALSE.			! since interactive
	ENDIF

* check to see if ylen or yorg has been changed
* if so, parameters saved prior to distorting yaxis are invalidated
        IF ( ylen .NE. temp_ylen ) ppl_saved_yaxlen = ylen
        IF ( yorg .NE. temp_yorg ) ppl_saved_yorg   = yorg

	RETURN

	END

