	SUBROUTINE DELETE_USER_VAR ( uvar )

* delete a user-defined variable 
* First locate each of its children and delete them

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450 7/97 - guts of routine moved to DELETE_USER_VAR_SUB

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	uvar

* internal variable declarations:
	INTEGER	uv, uv_parent

* delete child vars
	DO 200 uv = 1, max_uvar
	   IF ( uvar_num_items( uv ) .EQ. uvar_deleted ) GOTO 200
* ... delete if this variable is a child of uvar
	   IF (uvar_parent(uv).EQ.uvar) CALL DELETE_USER_VAR_SUB(uv)
 200	CONTINUE

* now delete the variable itself
        CALL DELETE_USER_VAR_SUB(uvar)
	      
	RETURN
	END
