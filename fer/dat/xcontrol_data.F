	BLOCK DATA XCONTROL_DATA

* initialization for control stack (CS)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/29/87

#ifdef unix
	include 'ferret_cmn/xcontrol.cmn'
#else
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
#endif

	DATA csp/ 0 /

	END
