	BLOCK DATA XGUI_DATA

* variables needed for GUI interactions

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V300:  6/93

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xgui.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XGUI.CMN'
#endif

	DATA pipe_open / .FALSE. /
	DATA gui_status	/ ferr_ok /
	DATA gui_fmt1 / '(I3, 1X, A4, 1x, A)' /
	DATA gui_char / '^' /    ! lead in for GUI-only commands

* in order to report newly created windows to the GUI:
	DATA gui_window_reported / max_gui_windows*.FALSE. /

	END
