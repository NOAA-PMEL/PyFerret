	SUBROUTINE PPL_SHASET( string )

* execute a PLOT+ SHASET command with the given string as argument

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230: 11/92 - for color map protection during viewport manipulations

#ifdef unix
	include 'ferret_cmn/plot_setup.parm'
!	include 'ferret_cmn/xprog_state.cmn'
!	include 'ferret_cmn/xplot_setup.cmn'
	include 'ferret_cmn/xrisc.cmn'		! to fix RISC FORTRAN bug
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
!	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
!	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
#endif

* calling argument declarations:
	CHARACTER*(*)	string

* internal variable declarations:

* initialize fixed length string for RISC (needed for DEC f77 v2.1)
	risc_buff = string
	len_rbuff = MIN( size_rbuff, LEN(string) )

* send the command to PLOT+  "SHASET [string]"
	CALL PPLCMD ( from, line, 0,
     .                'SHASET '//risc_buff(:len_rbuff), 1, 1 )

* always successful
        RETURN
	END
