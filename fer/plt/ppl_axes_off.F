	SUBROUTINE PPL_AXES_OFF(set_up)

* turn off the PLOT+ axes with the AXSET command but save the current
* state so the axes can be restored

* V450:  

#ifdef unix
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xcontrol.cmn'
	include 'tmap_pplv11inc/axis_inc.decl'
	include 'pplv11inc/AXIS.INC'		! variable iaxon
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	include 'FERRET_CMN:XCONTROL.CMN'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'
	INCLUDE 'PPLV11INC:AXIS.INC'	
#endif

* calling argument declarations
	LOGICAL set_up

* local variable declarations:
	INTEGER i
	CHARACTER buff*16

* static storage in COMMON
	INTEGER save_iaxon(4)
	COMMON /XAXIS_STATE/ save_iaxon

* save current axis state
	DO 10 i = 1, 4
 10	save_iaxon(i) = iaxon(i)

* turn all 4 axes off
	CALL PPLCMD ( from, line, 0, 'AXSET 0,0,0,0', 1, 1 )

* turn axis labels off
	CALL PPLCMD ( from, line, 0, 'XLAB', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YLAB', 1, 1 )

* if in /set_up mode this state will not be restored properly
* issue warning only if user typed in command (csp.eq.0)
	IF ( set_up .AND. csp.EQ.0 ) THEN
	   buff = ' '
	   WRITE (buff, '(4I3)' ) (save_iaxon(i), i=1, 4)
	   CALL WARN(
     .		'To restore axes after /SET_UP/NOAXES use PPL AXSET'
     .		//buff(:12) )
	ENDIF

	RETURN
	END
