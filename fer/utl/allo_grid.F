	SUBROUTINE ALLO_GRID( grid, status )

* allocate a scratch space in the grid COMMON to work on a grid definition
* Space is allocated in high grid subscripts working down

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/22/89
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xfr_grid.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'
#endif

* calling argument declarations
	INTEGER	grid, status

	grid = grd_stk_ptr - 1

	IF ( grid_name(grid) .NE. char_init16 ) CALL ERRMSG
     .		( ferr_stack_ovfl, status, 'grid stack',*5000 )

	grd_stk_ptr = grid

	status = ferr_ok
 5000	RETURN
	END
