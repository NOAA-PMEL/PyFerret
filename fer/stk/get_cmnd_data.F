	SUBROUTINE GET_CMND_DATA ( memory, dflt_cx, status )

* calculate grid(s) of data based on the expression given or implied.
* Expression(s) are in user-variable (uvar) structures.
* Implied expression is left from last command

* a sample command may look like:
*	prompt> CONTOUR/Z=50 TEMP[X=160E:160W]
* the "Z=50"		is a "command  qualifier" or "command  context modifier"
* the "X=160E:160W"	is a "variable qualifier" or "variable context modifier"
* contexts are transferred, modified and used in the following manner:
*   i)	skeletal context ( without grid-related info.) is specified in the call
*	Usually it is based on SET REGION,DTA_SET,etc. ( cx_last )
*  ii)	this context is modified by the command qualifiers given to produce a 
*	new skeletal context to be used for the evaluation of the current 
*	command ( cx_cmnd )
*  iii)	as the expression is evaluated each variable, as it is encountered,
*	begins with cx_cmnd and modifies it by the variable
*	qualifiers to produce the context at which the variable is to be
*	evaluated.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230:  7/27/92 - split into two separate routines, GET_PROT_CMND_DATA and
*                  and UUNPROTECT_CMND_DATA so that memory management can be
*                  used during command (e.g. LIST command needs to create
*                  permutations)
* V312: 5/94 - array "memory" as a calling argument

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations
	INTEGER	dflt_cx, status
	REAL	memory(*)

* local variable declarations
* ...

        CALL GET_PROT_CMND_DATA( memory, dflt_cx, status )
        IF ( status .NE. ferr_ok ) RETURN

        CALL UNPROTECT_CMND_DATA

	RETURN
	END



