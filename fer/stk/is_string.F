	SUBROUTINE IS_STRING( memory, *, *, status )

* create a string  "variable" in memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500:  1/99 - based on IS_CONST_VAR

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
	INTEGER	CGRID_SIZE,
     .		mr, cx


* --- end of introductory code ---

* WHEN THE TIME COMES TO RECOVER THIS STRING, HERE IS THE TECHNIQUE:
* ... note that the uvar number and item number of the string are
*     packed together in the cx_variable value
!	IF ( cx_variable(cx) .GT. 0 ) THEN
!	   uvar = cx_variable(cx)/1000
!	   item = cx_variable(cx) - 1000*uvar
!	   start = uvar_item_start( item, uvar )
!	   end   = uvar_item_end  ( item, uvar )
!	   string   =  uvar_text(uvar)(start:end)
!	ELSE
! ... for future grids of strings
!	   pointer to array of strings encoded in memory(k,blk)
!	   string = EXTRACT_FROM_STRING_GRID(memory(k,blk),i,j,k,l)
!	ENDIF

* initialize
	cx = is_cx( isp )

* reserve and catalog space for the result
* ... note 1: the size of the memory reserved will be equal to the size
*     of the underlying grid. For a string used as an argument in a function
*     call this will be a scalar (size=1). This is incomplete logic ...
*     headed for the eventual support of grids of strings, possibly computed
*     by functions (e.g. DATE_STRING(day,mon,yr,"format") )
* ... note 2: could create a lasting variable, but would need to make sure
*     that when DELETE_VARIABLE is called it also purges all cat_const_var
*     component variables. This approached seemed to add more complexity
*     than value ...
	CALL CREATE_TEMP_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( isp ) = mr

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
