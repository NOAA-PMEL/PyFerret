	SUBROUTINE XEQ_SPAWN

* SPAWN a subprocess to VMS DCL level

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/2/88
* v200: Unix/RISC port - 3/12/91 - coach for "^z" on Unix
* V230 6/92 - allow "system" calls for unix
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
#endif

# ifdef unix
	IF ( num_args .GT. 0 ) THEN
	   call system( cmnd_buff(arg_start(1):len_cmnd) )
	ELSE
* user can get choice of shells with "SPAWN csh", or whatever.
	   CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .    	' Use "^Z" for shell prompt.  Type "fg" to return.', 0)
	   CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .    	' Or type "SPAWN csh" (or shell of your choice) '
     .    	//'and "exit" to return', 0)
      ENDIF
#else
* VMS systems ...
* open PPLUS if it isn't yet
	IF ( .NOT.pplus_started ) THEN
	   CALL START_PPLUS
	ENDIF

* simply use the PPLUS SPAWN command
	CALL PPLCMD (' ',' ',0,'SPAWN',1,1)
#endif

	RETURN

	END

