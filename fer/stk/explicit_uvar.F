	LOGICAL FUNCTION EXPLICIT_UVAR( uvar )

* determine if the given "user-defined" variable was explicitly created (with
* DEFINE VARIABLE) or implicitly (e.g. with LIST U*V)


* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   8/8/89


#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	uvar

	EXPLICIT_UVAR = INDEX( uvar_name_code(uvar), '#' ) .EQ. 0

	RETURN
	END
