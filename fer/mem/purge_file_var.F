	SUBROUTINE PURGE_FILE_VAR( fvar )

* eliminate all stored mrs for the given file variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/23/89 - based on PURGE_USER_V*

#ifdef unix
	include		'ferret_cmn/ferret.parm'
	include 	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE		'FERRET_CMN:FERRET.PARM'
	INCLUDE 	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declaration:
	INTEGER	fvar

* internal variable declarations:
	INTEGER	mr

	DO 100 mr = 1, max_mrs
	   IF ( mr_protected( mr ) .EQ. mr_deleted ) GOTO 100
	   IF ( mr_variable ( mr ) .NE. fvar
     .	   .OR. mr_category ( mr ) .NE. cat_file_var )    GOTO 100

	   IF (  mr_protected( mr ) .NE. mr_not_protected
     .	   .AND. mr_protected( mr ) .NE. mr_perm_protected )
     .						STOP 'var prot err --> FVAR'

	   CALL DELETE_VARIABLE( mr )

 100	CONTINUE

	RETURN
	END
