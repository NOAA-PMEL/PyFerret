	SUBROUTINE WARN ( string )

* display a warning message to the operator

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  7/17/89
* Unix/RISC port "WARNING" --> "NOTE"
* V300: - 6/17/93: added GUI control option for messages
* V314: 8/22/94 *kob* IBM port - need to use risc buff for char. concat 
* V400: 11/94 - send warnings to GUI via 
* V320: 2/95 - use TMAP routine tm_note

#ifdef unix
#ifdef NO_PASSED_CONCAT
	include 'ferret_cmn/xrisc.cmn'
#endif
	include 'ferret_cmn/ferret.parm'
        include 'ferret_cmn/gui.parm'
	include 'ferret_cmn/xprog_state.cmn'
        include 'ferret_cmn/xgui.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
        INCLUDE 'FERRET_CMN:GUI.PARM'
        INCLUDE 'FERRET_CMN:XGUI.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	CHARACTER*(*)	string

* internal variable declarations:
	INTEGER TM_LENSTR1, llen

* initialize
	llen = TM_LENSTR1( string )

* let the TMAP library do the work
	CALL TM_NOTE( string(:llen), ttout_lun )

	RETURN
	END
