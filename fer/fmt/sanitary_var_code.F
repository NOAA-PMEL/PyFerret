	CHARACTER*(*) FUNCTION SANITARY_VAR_CODE ( category, variable )

* return name code for variable that is "sanitary" in the sense that it can
* be used as a part of a filename

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  12/4/89 - created to make more honest VAR_CODE for other purposes
* V230:  11/4/92 - expanded user variable name length

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER category, variable

* internal variable declarations:
	INTEGER		HASH_NAME
	CHARACTER	VAR_CODE*24, vcode*24

* get basic variable code
	vcode = VAR_CODE( category, variable )

* sanitize it
	IF ( INDEX(vcode,'#') .GT. 0 ) WRITE( vcode, 3000 ) 'E', 
     .				HASH_NAME( uvar_text( variable ), 999 )
 3000	FORMAT(A1,I3.3,4X)

* done
	SANITARY_VAR_CODE = vcode

	RETURN
	END
