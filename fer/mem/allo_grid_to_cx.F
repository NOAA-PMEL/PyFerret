	SUBROUTINE ALLO_GRID_TO_CX( cx, grid, status )

* Allocate a temporary grid from the grid stack (not a managed dynamic grid)
* and place the necessary flags about it in the given context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420 9/8/95 - for implicit grid management

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations
	INTEGER	cx, grid, status

* internal variable declarations

* allocate the grid slot
	CALL ALLO_GRID( grid,status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* initialize the grid to be completely unspecified
	CALL INIT_GRID(grid,
     .		'(implicit)',
     .		unspecified_int4 )


* place flags in the given context
	cx_has_impl_grid(cx) = .TRUE.
	cx_impl_grid(cx) = grid


* optional diagnostic message
#ifdef debug_dyn_grids
	IF ( mode_diagnostic )  CALL DIAG_OP
     .		('allocate', isact_class_grid, grid, 0)
#endif

 5000	RETURN
	END



