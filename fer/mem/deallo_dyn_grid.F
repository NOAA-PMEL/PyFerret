	SUBROUTINE DEALLO_DYN_GRID ( grid )

* deallocate the specified dynamic grid (no check made on grid number)
* and clear any memory references to this grid

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420 9/8/95 - new

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/interp_stack.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations
	INTEGER	grid

* local declarations
	INTEGER mr

* diagnostic message
	IF ( mode_diagnostic )  CALL DIAG_OP
     .	     ('dealloc ',isact_class_impl_grid,grid,0)

* deallocate the grid
	CALL TM_DEALLO_DYN_GRID(grid)

* if the grid was actually destroyed then delete any memory resident 
* variables that were built on this grid
	IF ( grid_use_cnt(grid) .EQ. 0 ) THEN
	   DO 100  mr = 1, max_mrs
	      IF ( mr_protected(mr) .NE. mr_deleted ) THEN
	         IF (mr_grid(mr) .EQ. grid) CALL DELETE_VARIABLE(mr)
	      ENDIF
 100	   CONTINUE
	ENDIF

	RETURN
	END
