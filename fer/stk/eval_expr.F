	SUBROUTINE EVAL_EXPR ( memory, dflt_cx, expression, status )

* "evaluate expression"
* calculate grid of data based on the expression given
* Saves (temporarily) expression in user-variable (uvar) structure.
* when done - return with the variable un-"protected" (flagged as deletable)

* a sample command may look like:
*	CONTOUR/Z=`TEMP[Z=0:1000@LOC:10]` TEMP[X=160E:160W]

* contexts are transferred, modified and used in the following manner:
*   i)	skeletal context ( without grid-related info.) is specified in the call
*	Usually it is based on SET REGION,DTA_SET,etc. ( cx_last )
*  ii)	this context is modified by the command qualifiers given to produce a 
*	new skeletal context to be used for the evaluation of the current 
*	command ( cx_cmnd )
*	the command context may, itself, contain grave-accented expressions
*	... those parts of the region are ignored until they are known
*  iii)	as the expression is evaluated each variable, as it is encountered,
*	begins with cx_cmnd and modifies it by the variable
*	qualifiers to produce the context at which the variable is to be
*	evaluated.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V400: 3/95 - based on GET_PROT_CMND_DATA

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xfr_grid.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'
#endif

* calling argument declarations
	INTEGER	dflt_cx, status
	REAL	memory(*) 
	CHARACTER*(*) expression

* local variable declarations
	INTEGER		cx_expr, mr
	CHARACTER	EXPR_NAME*8
	INTEGER		tlen

* initialize grid stack pointer (pushes down from top)
	tlen = LEN(expression)
	grd_stk_ptr = max_grids

* delete any memory left-overs from last command
	CALL DELETE_MRS_IN_PROGRESS

* * * * DETERMINE CONTEXT FOR COMMAND * * *
* ==> this needs to be replaced with a version that can evaluate incomplete
*	regions that have grave-accented region information
	CALL GET_NEW_CX( dflt_cx, cx_cmnd, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* store expression in the uvar list names as "EX#1"
	CALL DELETE_OLD_EXPR
	CALL BREAK_UP_EXPR( expression, num_uvars_in_cmnd, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	IF ( num_uvars_in_cmnd .NE. 1 ) CALL ERRMSG
     .			(ferr_invalid_command,status,expression,*5000)

* reserve a context slot to hold a fresh command context
	cx_stack_ptr = cx_stack_ptr + 1
	cx_expr = cx_stack_ptr

* evaluate the expression
	CALL TRANSFER_CONTEXT( cx_cmnd, cx_expr )
	CALL INTERP_STACK( memory, 1, cx_expr, EXPR_NAME(1), status )
	IF ( status .NE. ferr_ok ) GOTO 5000

! temp diagnostic
	   IF ( cx_stack_ptr .NE. cx_cmnd+1 ) WRITE (6,*) 'get_cmnd_???'

* flag this result as temporary/delete-able
        CALL UNPROTECT_CMND_DATA
	RETURN

* error exit
 5000	RETURN
	END
