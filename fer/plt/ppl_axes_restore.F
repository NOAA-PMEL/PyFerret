	SUBROUTINE PPL_AXES_RESTORE

* restore the PLOT+ axes after they have been turned off by PPL_AXES_OFF

* V450:  

#ifdef unix
	include 'ferret_cmn/plot_setup.parm'
	include 'tmap_pplv11inc/axis_inc.decl'
	include 'pplv11inc/AXIS.INC'		! variable iaxon
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'
	INCLUDE 'PPLV11INC:AXIS.INC'	
#endif

* calling argument declarations

* local variable declarations:
	INTEGER i
	CHARACTER buff*16

* static storage in COMMON
	INTEGER save_iaxon(4)
	COMMON /XAXIS_STATE/ save_iaxon

* convert 4 axis integers to character as arg to the AXSET command
	buff = ' '
	WRITE (buff, '(4I3)' ) (save_iaxon(i), i=1, 4)

* restore axis state
	CALL PPLCMD ( from, line, 0, 'AXSET '//buff, 1, 1 )

	RETURN
	END
