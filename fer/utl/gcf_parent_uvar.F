	INTEGER FUNCTION GCF_PARENT_UVAR( child_uvar )

* return the uvar number of the parent of this (child) uvar
* Note that the "TRUE" parent may be a grand-parent or great-grand
* For example, in the expression
*     A = FCN1(v1+v2) + B
* where FCN1 is a grid changing function then "FCN1(v1+v2)" will be a
* child of "A" and "v1+v2" will be a grand-child. The routine INIT_UVAR
* will have designated "A" as the parent of "v1+v2". The purpose of
* this routine would be to return the uvar of "A" when given the child_uvar
* whose definition is "v1+v2"

* V450: 7/7/97

* calling argument declarations
	INTEGER		child_uvar

* internal variable declarations
	INTEGER   uvar
	CHARACTER*12 name

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* initialize
	uvar = child_uvar

 100	CONTINUE
	IF (uvar_parent(uvar) .NE. 0) THEN
* ... get the parent of this uvar
	   uvar = uvar_parent(uvar)       ! a child becomes a parent
	   GOTO 100
	ELSE
* ... end of the line -- this uvar has no parent
	   GCF_PARENT_UVAR = uvar
	   RETURN
	ENDIF

	END
