	SUBROUTINE PURGE_USER_VAR( uvar )

* eliminate all stored mrs for the given user defined variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/23/87
* revision 0.1 - 9/30/87 - added mr_perm_prot
* V200:  6/9/89 - deleted variables flagged in mr_protected

#ifdef unix
	include		'ferret_cmn/ferret.parm'
	include 	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE		'FERRET_CMN:FERRET.PARM'
	INCLUDE 	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declaration:
	INTEGER	uvar

* internal variable declarations:
	INTEGER	mr

	DO 100 mr = 1, max_mrs
	   IF ( mr_protected( mr ) .EQ. mr_deleted ) GOTO 100
	   IF ( mr_variable ( mr ) .NE. uvar
     .	   .OR. mr_category ( mr ) .NE. cat_user_var )    GOTO 100

	   IF (  mr_protected( mr ) .NE. mr_not_protected
     .	   .AND. mr_protected( mr ) .NE. mr_perm_protected )
     .						STOP 'var prot err --> UVAR'

	   CALL DELETE_VARIABLE( mr )

 100	CONTINUE

	RETURN
	END

