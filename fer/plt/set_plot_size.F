	SUBROUTINE SET_PLOT_SIZE( xsize, ysize )

* adjust PPLUS SIZE command to re-size plot

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 9/16/88 
* Unix/RISC port - 10/9/91 - unsave previous axis lengths held by
*                            routine setup_keys
*                          - also restore the origin
* V230:  11/5/92 - removed axis resizing to SET_AX_SIZES
 
#ifdef unix
	include 'ferret_cmn/plot_setup.parm'
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
#endif

* calling argument declarations:
	REAL	xsize, ysize

* internal variable declarations:
	CHARACTER buff*24

* tell it to PPLUS
	WRITE ( buff, 3000 ) xsize, ysize
	CALL PPLCMD ( from, line, 0, 'SIZE '//buff, 1, 1 )
 3000	FORMAT(2F12.3)

	RETURN
	END
