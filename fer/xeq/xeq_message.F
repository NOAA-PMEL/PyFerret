	SUBROUTINE XEQ_MESSAGE

* execute the MESSAGE command

* MESSAGE text of message	- displays message, prompts for <CR> and waits
* MESSAGE/QUIET text		- displays message and waits for <CR>
* MESSAGE/CONTINUE text		- displays message and continues

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/10/86
* V230:  7/30/92 - bug fix: avoid final quote ("hello" comes out 'hello"' now)
*        8/11/92 - allow blank messages (blank line on output)
*       12/15/92 - pause w/ TM_FRIENDLY_READ so SGI's w/out backing store work
*                - and fix non-functioning of /QUIET on Unix (was .NOT.integer)
* V300:  4/22/93 - fix treatment of enclosing quotes with ALL_1_ARG
*	 6/16/93 - different prompts, etc. for GUI control
*		 - interrupt if GUI interrupt sequence is received
* V314	8/25/94 *kob* IBM port - need to correct syntax error
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	 2/28/95 - SPLIT_GUI_MESSAGE --> TM_SPLIT_MESSAGE
* Linux Port - *kob* 12/96 - had to put single quotes around hello 
*			     in comment up above - linux f90 freaked out.
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
        include 'ferret_cmn/xrisc.cmn'
        include 'ferret_cmn/xgui.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
        INCLUDE 'FERRET_CMN:XRISC.CMN'
        INCLUDE 'FERRET_CMN:XGUI.CMN'
#endif

* local parameter declarations
	INTEGER		slash_continue, slash_quiet
	PARAMETER     (	slash_continue	  = 1,
     .			slash_quiet	  = 2 )
	
* local variable declarations
	LOGICAL fexist
	INTEGER status

* treat all text as a single argument
	CALL ALL_1_ARG

* write the message if one has been supplied
	IF ( num_args .EQ. 1 ) THEN
	   CALL TM_SPLIT_MESSAGE( ttout_lun,
     .				cmnd_buff(arg_start(1):arg_end(1)) )
        ELSEIF ( qual_given(slash_continue) .GT. 0
     .	 .AND.  .NOT.mode_gui ) THEN
              WRITE ( ttout_lun, * ) 
        ENDIF
* continue or wait ?
	IF ( qual_given( slash_continue ) .GT. 0
     .	.OR. mode_gui ) RETURN

* prompt for <CR> ?
* (in gui mode ferret_dispatch will send a flag back to the GUI)
	IF ( qual_given( slash_quiet ) .EQ. 0 ) THEN
	   WRITE ( ttout_lun, * ) ' Hit Carriage Return to continue'
	ENDIF

* wait for continue signal (use TM_FRIENDLY_READ 12/92)
        CALL TM_FRIENDLY_READ(' ', risc_buff, *1000 )

* check for interrupt character signal "^>"
 1000	IF ( risc_buff(1:1) .EQ. gui_char
     . .AND. risc_buff(2:2) .EQ. '>' )  CALL ERRMSG( ferr_interrupt,
     .						     status, ' ', *2000 )

 2000	RETURN
	END
