	SUBROUTINE IS_GRID_MODE( status )

* set up interp stack to begin determining the grid of an algebraic expression

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/28/89
* V420:  1/95 - need to init cx_delta since it is used by dynamic axes

* calling argument declarations:
	INTEGER	status

* internal variable declarations:
	INTEGER cx, big_cx, idim

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

* carve a space on the context stack for a "bland" (1:1,1:1,1:1,1:1) context
	cx = cx_stack_ptr
	cx_grid(cx) = unspecified_int4		! for pseudo_vars in pass 1
	DO 100 idim = 1, 4
	  cx_lo_ss(cx,idim) = 1
	  cx_hi_ss(cx,idim) = 1
	  cx_by_ss(idim,cx) = .TRUE.
	  cx_trans(idim,cx) = trans_no_transform
	  cx_given(idim,cx) = .FALSE.
	  cx_delta(idim,cx) = unspecified_val8
 100	CONTINUE

* diagnostic alert
	IF ( mode_diagnostic )
     .			CALL DIAGNOSTIC_OUT( 'getgrid', cx, point_to_cx )

* initialize a level on the interpretion stack for the activity
	is_phase( isp ) = 0

* allocate another context for the "merged" grid
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	big_cx = cx_stack_ptr
	is_big_cx( isp ) = big_cx
	CALL ALLO_GRID( cx_grid(big_cx), status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* set flag indicating that the first component grid has not yet been encountered
	is_sum( isp ) = isalg_transfer
	RETURN

* error exit
 5000	RETURN
	END	

