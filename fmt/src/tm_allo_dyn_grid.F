	SUBROUTINE TM_allo_dyn_grid( grid, status )

* Allocate a grid from the dynamic grid area (above max_grids)

* The grids in xtm_grid.cmn fall into three categories (ideally they could
* and should all be category 3, but the development was historical)
* category 1: explicit (static) grids
*	these are located in the low indices of the grid common
*	(e.g. grid_name(3)).  They are created by READ_GRIDFILE, CD_*_GRIDS,
*	and by DEFINE GRID inside Ferret.  These are the grids that are
*	returned by TM_FIND_GRID_SLOT.
* category 2: temporary grids
*	these are located beginning at max_grids-1 and descending.  They have
*	only very temporary existence during the GET_UVAR_GRID process -- as
*	holding areas while dynamic grids (category 3) are built.
* category 3: dynamic grids
*	these are in the upper indices: above max_grids.  They are managed
*	efficiently with linked lists and usage counts.  They are used by
*	Ferret to hold "implicit" grids -- implied by expressions such as
*	SALT[GX=U] (the SALT grid with its X axis replaced from the U grid).
*	They are deleted as necessary to ensure that new grid definitions and
*	changes in user variable definitions do not corrupt memory. 

* Steve Hankin  NOAA/PMEL/TMAP

* 9/13/95 for Ferret version 4.2

* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	include 'tmap_format/xio.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
#endif

* Argument definitions
	INTEGER grid, status

* Local definitions
	INTEGER next_free_grid

* take a grid from the "free" list and put it at the start of the "used" list
	IF ( grid_free_ptr .NE. 0 ) THEN
* ... unhook from free list
	   grid = grid_free_ptr
	   next_free_grid  	= grid_flink(grid)
	   grid_free_ptr	= next_free_grid
* ... hook onto front of used list
	   grid_flink(grid)	= grid_flink(max_grids)
	   grid_blink(grid)	= max_grids
	   grid_flink(max_grids)= grid
	   grid_blink(grid_flink(grid))	= grid
	ELSE
	   CALL TM_ERRMSG (merr_gridlim, status,
     .                  'TM_ALLO_DYN_GRID', no_descfile, no_stepfile,
     .                  '>num_dyn_grids dynamic grids',
     .                  no_errstring, *9999)
	ENDIF

* signal that it is in use by a single claimant
	grid_use_cnt(grid) = 1

* successful completion
	status = merr_ok
 9999	RETURN

	END
