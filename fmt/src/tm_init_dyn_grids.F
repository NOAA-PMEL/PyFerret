	SUBROUTINE TM_init_dyn_grids

* Initialize machinery for dynamic grids and lines

* Steve Hankin  NOAA/PMEL/TMAP
*
* 9/13/95 for Ferret version 4.2

* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/implicit.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:IMPLICIT.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* Argument definitions

* Local definitions
	INTEGER i


***** GRIDS *****
* grid(max_grids) is used as the chain start for the "in-use" grids
	grid_flink(max_grids) = max_grids	! points to self: empty chain

* no "dynamic" grids (above max_grids) are "used" (yet)
	DO 100 i = 1, max_dyn_grids
 100	grid_use_cnt(i) = 0

* all the "dynamic" grids (eq to or above max_grids) in the "free" linked list
	grid_free_ptr = max_grids+1
	DO 200 i = max_grids+1, max_dyn_grids
 200	grid_flink(i) = i+1
	grid_flink(max_dyn_grids) = 0	! end of chain

***** LINES *****
* line(max_lines) is used as the chain start for the "in-use" lines
	line_flink(max_lines) = max_lines	! points to self: empty chain

* no "dynamic" lines (above max_lines) are "used" (yet)
	DO 300 i = 1, max_dyn_lines
	   line_use_cnt  (i) = 0
	   line_has_edges(i) = .FALSE.
 300	CONTINUE
	DO 310 i = max_lines+1, max_dyn_lines
	   line_parent(i) = 0
	   line_class (i) = pline_class_basic
 310	CONTINUE

* all the "dynamic" lines (eq to or above max_lines) in the "free" linked list
	line_free_ptr = max_lines+1
	DO 400 i = max_lines+1, max_dyn_lines
 400	line_flink(i) = i+1
	line_flink(max_dyn_lines) = 0	! end of chain

	RETURN

	END


