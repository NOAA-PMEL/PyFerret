	SUBROUTINE TM_use_line( line )

* increment the usage count on a line
* see docs in tm_allo_dyn_line

* Steve Hankin  NOAA/PMEL/TMAP

* 10/3/95 for Ferret version 4.2

* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* Argument definitions
	INTEGER line

* Local definitions

	line_use_cnt(line) = line_use_cnt(line) + 1

* always successful completion
	RETURN

	END
