	SUBROUTINE DEALLO_ALL_GRIDS

* deallocate all scratch space from the grid COMMON stack
* (9/95) also deallocate any dynamic grids that were in progress
* (used following error conditions)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/22/89
* V230:  7/24/92 - bug fix: loop index (igrid)  wasn't used inside loop
*                           grd_stk_ptr was!
* V420:  9/20/95 - deallocate dynamic grids, too
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/interp_stack.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xfr_grid.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'
#endif

* internal variable declarations
	INTEGER	igrid, i, cx, status


* deallocate the temporary grids from the upper grid stack
* (see documentation in tm_allo_dyn_grid)
* this may occasionally also deallocate associated dynamic axes
	DO 100 igrid = grd_stk_ptr, max_grids-1
	   CALL DEALLO_GRID(status)
	   IF ( status .NE. ferr_ok ) THEN	
	      grd_stk_ptr = max_grids
	      GOTO 150			! internal error!!
	   ENDIF
 100	CONTINUE

* Deallocate any dynamic grid (and associated memory vars) the need for which
* has been terminated by this error condition.
* A record of all of the dynamic grids in progress is implicitly available
* on the interpretation stack.
 150 	DO 200 cx = cx_stack_ptr, cx_stack_ptr_base, -1
	   IF ( cx_has_impl_grid(cx) .AND. cx_grid(cx).GT.max_grids) THEN
	      CALL DEALLO_DYN_GRID(cx_grid(cx))
	      cx_has_impl_grid(cx) = .FALSE.
	   ENDIF
 200	CONTINUE

	RETURN
	END
