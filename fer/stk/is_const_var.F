	SUBROUTINE IS_CONST_VAR( memory, *, *, status )

* create a constant valued array  "variable" in memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500: 12/98 - based on IS_CONSTANT

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
	INTEGER	CGRID_SIZE,
     .		mr, cx, size, uvar, item, start, end, n

* --- end of introductory code ---
* initialize
	cx = is_cx( isp )
	size = CGRID_SIZE(cx)

* reserve and catalog space for the result
* ... note: could create a lasting variable, but would need to make sure
*     that when DELETE_VARIABLE is called it also purges all cat_const_var
*     component variables. This approached seemed to add more complexity
*     than value ...
	CALL CREATE_TEMP_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( isp ) = mr

* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('const_v',mr,point_to_mr)

* parse and insert the constant array
* ... note that the uvar number and item number of "{1,3,5}" are
*     packed together in the cx_variable value
	uvar = cx_variable(cx)/1000
	item = cx_variable(cx) - 1000*uvar
	start = uvar_item_start( item, uvar )
	end   = uvar_item_end  ( item, uvar )

! ... as-of 12/98 limited to one block max array lentgh
	CALL PARSE_NUMBER_LIST(uvar_text(uvar)(start:end),
     .			       memory( 1, mr_blk1(mr)),
     .			       size,
     .			       n,
     .			       status)
	IF ( status .NE. ferr_ok ) GOTO 5000

* set the bad data flag
	mr_bad_data( mr ) = bad_val4

* got the requested component
	RETURN 2
! RETURN 1, which is to request a sub-component, is never used ...

* error exit
 5000	RETURN
	END	
