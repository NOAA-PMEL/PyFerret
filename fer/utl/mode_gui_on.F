	SUBROUTINE MODE_GUI_ON

* turn on MODE GUI for gui controll

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V320: 9/94 
*	4/95 - added alt_messages

	include 'tmap_format/xalt_messages.cmn_text'
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'

	mode_state(pmode_gui,1) = .TRUE. 
	alt_messages = .TRUE.

	RETURN
	END

