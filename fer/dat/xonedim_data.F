	BLOCK DATA XONEDIM_DATA

* variable declarations for COMMON/ XONEDIM /
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/17/86

#ifdef unix
	include 'ferret_cmn/gfdl.parm'
	include 'ferret_cmn/xonedim.cmn'
#else
	INCLUDE 'FERRET_CMN:GFDL.PARM'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'
#endif

* lines stolen from Pacanowski's SUBROUTINE GRIDS
      DATA DZ/10*10.E2,12.5E2,15.E2,17.5E2,20.E2,25.E2,30.E2,40.E2,
     * 57.E2,91.E2,151.E2,242.E2,357.E2,475.E2,566.E2,3*650.E2/

	END
