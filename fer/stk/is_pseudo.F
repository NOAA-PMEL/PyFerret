	SUBROUTINE IS_PSEUDO( memory, *, status )

* create a gridded "pseudo-variable" (eg. "X" or "K")

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/2/87
* V200:  6/26/89 - 4-D symmetric version
*	10/16/89 - changed adj. array bounds for MAKE_PSEUDO_VAR call
*	11/30/89 - added diagnostic output
* V312: 5/94 - array "memory" as a calling argument

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	status
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	INTEGER	mr, cx, idim

* --- end of introductory code ---
* initialize
	cx = is_cx( isp )

* remove regridding flag 
* for pseudo-vars it just means the grid was explicitly specified - there
* is no regridding)
	cx_unstand_grid( cx ) = .FALSE.

* reserve and catalog space for the result
	CALL CREATE_TEMP_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( isp ) = mr

* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('pseudo ',mr,point_to_mr)

* fill the space with the desired data
	idim = MOD( cx_variable(cx)-1, 4 ) + 1
	CALL MAKE_PSEUDO_VAR( cx, idim, memory(1, mr_blk1(mr)), mr,
     .			      mr_lo_ss(mr,idim), mr_hi_ss(mr,idim) )

	RETURN 1

* error exit
 5000	RETURN
	END	
