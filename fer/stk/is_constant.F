	SUBROUTINE IS_CONSTANT( memory, *, status )

* create a constant value "variable" in memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/28/87
* revision 0.1 - 6/23/88 - set bad data flag to -2.1E34 so -1.E34 is valid
* V200:  6/26/89 - just a single value
*	11/30/89 - added diagnostic output
* V312: 5/94 - array "memory" as a calling argument

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	status
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	INTEGER	mr, cx

* equivalences
	REAL	cx_value( max_context )
	EQUIVALENCE ( cx_variable, cx_value )

* --- end of introductory code ---
* initialize
	cx = is_cx( isp )

* reserve and catalog space for the result
	CALL CREATE_TEMP_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( isp ) = mr

* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('constan',mr,point_to_mr)

* insert the constant
	memory( 1, mr_blk1(mr) ) = cx_value(cx)

* set the bad data flag to an unusual number
	mr_bad_data( mr ) = -2.1E34

	RETURN 1

* error exit
 5000	RETURN
	END	
