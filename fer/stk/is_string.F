	SUBROUTINE IS_STRING( memory, *, *, status )

* create a string  "variable" in memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500:  1/99 - based on IS_CONST_VAR
* V530:  8/00 *sh* - support for string arrays -- use dynamic memory alloc

#ifdef unix
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xprog_state.cmn'
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
	INTEGER	MGRID_SIZE, mr, cx, uvar, item, start, end

* --- end of introductory code ---

* NOTE: it is by historical accident that this routine exists at all.
* To be fully consistent the string variables created by this routine should
* be created by is_constant with a cx_type of ptype_string and a cx_category
* of cat_constant. So it goes ...

* This routine extracts the sring given in a user variable definition and
* inserts it into a Ferret string variable. The string, itself, is stored
* in dynamic storage via C. The pointer to the string is put into the
* Ferret mr storage.

* initialize
	cx = is_cx( isp )

* reserve and catalog space for the pointer(s) to the string
	CALL CREATE_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( isp ) = mr

* save the pointer array address in cx_c_pointer
	CALL INIT_C_STRING_ARRAY( MGRID_SIZE(mr),
     .				  memory(1,mr_blk1(mr)),
     .				  mr_c_pointer(mr)      )

* determine the string to be stored
	uvar = cx_variable(cx)/1000
	item = cx_variable(cx) - 1000*uvar
	start = uvar_item_start( item, uvar ) + 1  ! +1 skip quote
	end   = uvar_item_end  ( item, uvar ) - 1  ! -1 skip quote

* save the string in dynamic memory
	CALL STORE_STRING(uvar_text(uvar)(start:end),
     .			  mr, 0, status)
	IF (status .NE. ferr_ok) GOTO 5000

* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('string',mr,point_to_mr)

* set the bad data flag
	mr_bad_data( mr ) = bad_val4

* got the requested component
	status = ferr_ok
	RETURN 2
! RETURN 1, which is to request a sub-component, is never used ...

* error exit
 5000	RETURN
	END	
