	SUBROUTINE CLEAN_STACK

* following an error during stack computation restore the stack variables
* to a "clean" state

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* Unix/RISC port - version 0.0 - 9/25/91 *sh*
* this corrects a bug in the VMS version, too: SET GRID following an error

#ifdef unix
	include		'ferret_cmn/ferret.parm'
	include 	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE		'FERRET_CMN:FERRET.PARM'
	INCLUDE 	'FERRET_CMN:XVARIABLES.CMN'
#endif

* unset is_cx(1) reflecting last data access as a failure
      is_cx(1) = unspecified_int4

	RETURN
	END

