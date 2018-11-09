	SUBROUTINE IS_CONST_VAR(  *, *, status )

* create a constant valued array  "variable" in memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500: 12/98 - based on IS_CONSTANT
* V530:  8/00 *sh* - allow lists of strings
* V533:  7/01 *sh* - allow single quoted strings
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V693+ *sh* 2/15 - support for _DQ_ and _SQ_ as quotation indicators
 
        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xprog_state.cmn'

* calling argument declarations:
	INTEGER	status

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

* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('const_v',mr,point_to_mr)

* parse and insert the constant array
* ... note that the uvar number and item number of "{1,3,5}" are
*     packed together in the cx_variable value
	uvar = cx_variable(cx)/1000
	item = cx_variable(cx) - 1000*uvar
	start = uvar_item_start( item, uvar )
	end   = uvar_item_end  ( item, uvar )

	IF ( INDEX(uvar_text(uvar)(start:end), '"')    .GT. 0
     .	.OR. INDEX(uvar_text(uvar)(start:end), "'")    .GT. 0
     .	.OR. INDEX(uvar_text(uvar)(start:end), "_DQ_") .GT. 0
     .	.OR. INDEX(uvar_text(uvar)(start:end), "_SQ_") .GT. 0 ) THEN
* ... set up FORTRAN-independent pointer
	   CALL INIT_C_STRING_ARRAY( size,
     .				     memry(mr)%ptr,
     .				     mr_c_pointer(mr)      )

* ... array of strings
* NOTE: this routine will alter the context size and replace the mr
* (allocating a larger block of memory) if the SYSTEM: syntax is used
	   CALL PARSE_STRING_LIST(cx,
     .				  uvar_text(uvar)(start:end),
     .			          mr,
     .			          status)
	ELSE
* ... array of numbers
! ... as-of 12/98 limited to one block max array lentgh
	   CALL PARSE_NUMBER_LIST(uvar_text(uvar)(start:end),
     .			          memry(mr)%ptr,
     .			          size,
     .			          n,
     .			          status)
	ENDIF
	IF ( status .NE. ferr_ok ) GOTO 5000

* set the bad data flag
	mr_bad_data( mr ) = bad_val4

* result is ready to return to the stack
	is_mr( isp ) = mr

* got the requested component
	RETURN 2
! RETURN 1, which is to request a sub-component, is never used ...

* error exit
 5000	RETURN
	END	
