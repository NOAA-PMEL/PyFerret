	SUBROUTINE GCF_CHILD_VAR_NAME( parent_uvar, char1, name )

* return the (hidden) name of a child variable constructed from
* the number of its parent variable and the start character of the
* grid-changing reference within the parent definition

* For example, in the definition
*     yes? LET var = 2 * GC_FUNCT( a,b,c )
* the string "GC_FUNCT( a,b,c )" would become a child variable with a name
* based on the uvar number of "var" and a start char of 5 (following "2 * ")

* V450: 12/96
*        7/7/97 - revert to one name in all cases -- use DIAGNOSTIC_OUT
*                 to make diagnostic output more readable

* calling argument declarations
	INTEGER		parent_uvar, char1
	CHARACTER	name*(*)

* internal variable declarations

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* sanity check
	IF ( LEN(name) .LT. 11 ) THEN
	   STOP 'GCF_CHILD_VAR_NAME'
	ENDIF

!	IF ( char1.LT.100 .AND. parent_uvar.LT.100) THEN
!	   WRITE (name, 3000) char1, parent_uvar
! 3000	   FORMAT ('(C',I2.2,',V',I2.2,')')	! e.g. "(C12,V02)"
!	ELSE
	   WRITE (name, 3100) char1, parent_uvar
 3100	   FORMAT ('(C',I3.3,',V',I3.3,')')	! e.g. "(C112,V002)"
!	ENDIF

	RETURN
	END
